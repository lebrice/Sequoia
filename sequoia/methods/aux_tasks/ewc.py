"""Elastic Weight Consolidation as an Auxiliary Task.

This is a simplified version of EWC, that only currently uses the L2 norm, rather
than the Fisher Information Matrix.

TODO: If it's worth it, we could re-add the 'real' EWC using the nngeometry
package, (which I don't think we need to have as a submodule).
"""

from copy import deepcopy
from collections import deque
from dataclasses import dataclass
from typing import (Dict, Iterable, Iterator, List, Mapping, MutableMapping,
                    Optional, Tuple)

import torch
from simple_parsing import choice
from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader, TensorDataset, Dataset

# from sequoia.common.dict_buffer import DictBuffer
from sequoia.common.loss import Loss
from sequoia.methods.ewc_method import FIM
from sequoia.methods.aux_tasks.auxiliary_task import AuxiliaryTask
from sequoia.methods.models.output_heads import OutputHead
from sequoia.utils import dict_intersection, dict_union
from sequoia.utils.logging_utils import get_logger

from nngeometry.generator.jacobian import Jacobian
from nngeometry.layercollection import LayerCollection
from nngeometry.object.pspace import (PMatAbstract, PMatKFAC, PMatDiag,
                                      PVector)

from sequoia.settings.base.objects import Observations
from typing import ClassVar, Dict, Optional, Type, TypeVar, Union, List
from torch.autograd import Variable


logger = get_logger(__file__)


class EWCTask(AuxiliaryTask):
    """ Elastic Weight Consolidation, implemented as a 'self-supervision-style' 
    Auxiliary Task.

    some functions are taken from this repo: https://github.com/kuc2477/pytorch-ewc/tree/master
    """
    name: str = "ewc"

    @dataclass
    class Options(AuxiliaryTask.Options):
        """ Options of the EWC auxiliary task. """

        #Batchsize to be used when computing FIM
        batch_size_fim: int = 64
        # Number of observations to use for FIM calculation
        sample_size_fim: int = 400
        #Fisher information type  (diagonal or block diagobnal)
        fim_representation: PMatAbstract = choice({'diagonal':PMatDiag, 'block_diagonal':PMatKFAC}, default=PMatKFAC)

    def __init__(self,
                 *args,
                 name: str = None,
                 options: "EWC.Options" = None,
                 method: str = 'generic',
                 **kwargs):
        super().__init__(*args, options=options, name=name, **kwargs)
        self.options: EWCTask.Options
        self.previous_task: int = None
        self._i: int = 0
        self.n_switches: int = 0
        self.method = method if method in ['dqn', 'a2c'] else 'generic'
        logger.info(f'EWC will be applied in {self.method} mode.')

        self.FIMs: List[PMatAbstract] = None
        self.previous_model_weights: PVector = None
        self.FIM_representation = self.options.fim_representation
        self.observation_collector = deque(maxlen=self.options.sample_size_fim)

    def estimate_fisher(self, dataset, sample_size, batch_size=32):
        # sample loglikelihoods from the dataset.
        data_loader = DataLoader(dataset, batch_size)
        loglikelihoods = []
        for x in data_loader:
            # x.values() - since x returned from uterating over the Dataloader is a dictionary.
            observation = self.observation_collector[0].__class__(*x.values())
            loglikelihoods.append(
                torch.log_softmax(self._model(observation).logits, dim=1).max(1).values #[range(batch_size), y.data]
            )
            if len(loglikelihoods) >= sample_size // batch_size:
                break
        # estimate the fisher information of the parameters.
        loglikelihoods = torch.cat(loglikelihoods).unbind()
        loglikelihood_grads = zip(*[torch.autograd.grad(
            l, self._shared_net.parameters(),
            retain_graph=(i < len(loglikelihoods)), allow_unused=True,
        ) for i, l in enumerate(loglikelihoods, 1)])
        #here an error might be raised, in case grads dont backprop into the
        loglikelihood_grads = [torch.stack(gs) for gs in loglikelihood_grads]
        fisher_diagonals = [(g ** 2).mean(0) for g in loglikelihood_grads]
        param_names = [
            n.replace('.', '__') for n, p in self._shared_net.named_parameters()
        ]
        return {n: f.detach() for n, f in zip(param_names, fisher_diagonals)}

    def consolidate(self, fisher):
        for n, p in self._shared_net.named_parameters():
            n = n.replace('.', '__')
            self._model.register_buffer('{}_mean'.format(n), p.data.clone())
            self._model.register_buffer('{}_fisher'
                                 .format(n), fisher[n].data.clone())
    def on_task_switch(self, task_id: Optional[int]): #, dataloader: DataLoader, method: str = 'a2c')-> None:
        """ Executed when the task switches (to either a known or unknown task).
        """
        if not self.enabled:
            return
        logger.info(f"On task switch called: task_id={task_id}")

        if self._shared_net is None:
            logger.info(f"On task switch called: task_id={task_id}, EWC can not be applied as the used net has no shared part.")

        #TODO: deal with situation when task_id is None
        elif self.previous_task is None and self.n_switches == 0 and not task_id:
            self.previous_task = task_id
            logger.info("Starting the first task, no EWC update.")
            self.n_switches += 1
        
        elif task_id is None or task_id > self.previous_task and self._model.training:
            #we dont want to go here at test tiem
            # NOTE: We also switch between unknown tasks.
            logger.info(f"Switching tasks: {self.previous_task} -> {task_id}: "
                         f"Updating the EWC 'anchor' weights.")
            self.previous_task = task_id
            self.previous_model_weights = PVector.from_model(self._shared_net).clone().detach()

            def splitbatch(x: List[Observations]) -> List[Observations]:
                """ Splits a list of batched observatins into a list of observations with batch size 1
                """
                res = []
                for obs in x:
                    res +=obs.split()
                return res
            new_fim = self.estimate_fisher(splitbatch(self.observation_collector), self.options.sample_size_fim)
            self.consolidate(new_fim) #,task=self.previous_task)
            self.n_switches += 1
            self.observation_collector = deque(maxlen=self.options.sample_size_fim)

    def ewc_loss(self):
        try:
            losses = []
            for n, p in self._shared_net.named_parameters():
                # retrieve the consolidated mean and fisher information.
                n = n.replace('.', '__')
                mean = getattr(self._model, '{}_mean'.format(n))
                fisher = getattr(self._model, '{}_fisher'.format(n))
                # wrap mean and fisher in variables.
                mean = Variable(mean)
                fisher = Variable(fisher)
                # calculate a ewc loss. (assumes the parameter's prior as
                # gaussian distribution with the estimated mean and the
                # estimated cramer-rao lower bound variance, which is
                # equivalent to the inverse of fisher information)
                losses.append((fisher * (p-mean)**2).sum())
            return sum(losses)
        except AttributeError:
            # ewc loss is 0 if there's no consolidated parameters.
            return (
                Variable(torch.zeros(1)).to(self.device)
            )
    @property
    def _shared_net(self):
        """
        Returns 'None' if there is not shared network part, othervise returns the shared net
        """
        if self._model.encoder is None:
            return None
        elif isinstance(self._model.encoder, Iterable):
            if len(self._model.encoder)==0:
                return None
        return self._model.encoder

    def get_loss(self, x: Union[Tensor, Observations], *args, **kwargs) -> Loss:
        """Gets the 'EWC' loss. 
        """
        if self._model.training:
            if isinstance(x, Observations):  
                self.observation_collector.append(x)
            elif isinstance(x, Tensor):
                self.observation_collector.append(x)
            else:
                self.observation_collector.append(x.observations)
        if self.previous_task is None or not self.enabled or self._shared_net is None:
            # We're in the first task: do nothing.
            return Loss(name=self.name)        
        loss = self.ewc_loss()
        self._i += 1
        ewc_loss = Loss(
            name=self.name,
            loss=loss,
        )
        return ewc_loss

class L2ParamReg(AuxiliaryTask):
    """ EWC like regularization using simple with L2 distance, implemented as a 'self-supervision-style' 
    Auxiliary Task.
    """
    name: str = "l2paramregularizer"

    @dataclass
    class Options(AuxiliaryTask.Options):
        """ Options of the EWC auxiliary task. """
        # Wether to use the absolute difference of the weights or the difference
        # in the `regularize` method below.
        use_abs_diff: bool = False
        # The norm term for the 'distance' between the current and old weights.
        distance_norm: int = 2

    def __init__(self,
                 *args,
                 name: str = None,
                 options: "EWC.Options" = None,
                 **kwargs):
        super().__init__(*args, options=options, name=name, **kwargs)
        self.options: EWCTask.Options
        self.previous_task: int = None
        # TODO: Figure out a clean way to persist this dict into the state_dict.
        self.previous_model_weights: Dict[str, Tensor] = {}
        self._i: int = 0
        self.n_switches: int = 0

    def state_dict(self, *args, **kwargs) -> Dict:
        state = super().state_dict(*args, **kwargs)
        state.update(self.previous_model_weights)
        return state

    def load_state_dict(self, state_dict: Dict[str, Tensor], strict: bool = True) -> Tuple[List[str], List[str]]:
        missing: List[str]
        unexpected: List[str]
        missing, unexpected = super().load_state_dict(state_dict=state_dict, strict=False)
        if unexpected and not self.previous_model_weights:
            # Create the previous model weights, if needed.
            self.previous_model_weights.update(deepcopy({
                k: v.detach() for k, v in self.model.named_parameters()
            }))
        # TODO: Make sure that the model itself (i.e. its output heads, etc) gets
        # restored before this here.
        for key in unexpected.copy():
            if key in self.previous_model_weights:
                # Update the value in the 'previous model weights' dict.
                self.previous_model_weights[key] = state_dict[key]
                unexpected.remove(key)
        return missing, unexpected

    def disable(self):
        """ Disable the EWC loss. """
        # save a little bit of memory by clearing the weights.
        self.previous_model_weights.clear()
        return super().disable()

    def enable(self):
        # old_weights = parameters_to_vector(self.model.parameters())
        # self.register_buffer("old_weights", old_weights, persistent = True)
        return super().enable()

    def on_task_switch(self, task_id: int)-> None:
        """ Executed when the task switches (to either a new or known task).
        """
        if not self.enabled:
            return
        if self.previous_task is None and self.n_switches == 0:
            logger.debug(f"Starting the first task, no EWC update.")
            pass
        elif task_id is None or task_id != self.previous_task:
            logger.debug(f"Switching tasks: {self.previous_task} -> {task_id}: "
                         f"Updating the EWC 'anchor' weights.")
            self.previous_task = task_id
            self.previous_model_weights.clear()
            self.previous_model_weights.update(deepcopy({
                k: v.detach() for k, v in self.model.named_parameters()
            }))
            # self.old_weights = parameters_to_vector(self.model.parameters())
        self.n_switches += 1

    def get_loss(self, *args, **kwargs) -> Loss:
        """Gets the loss. 

        NOTE: This is a simplified version of EWC where the loss is the P-norm
        between the current weights and the weights as they were on the begining
        of the task.

        This doesn't actually use any of the provided arguments.
        """
        if self.previous_task is None:
            # We're in the first task: do nothing.
            return Loss(name=self.name)

        old_weights: Dict[str, Tensor] = self.previous_model_weights
        new_weights: Dict[str, Tensor] = dict(self.model.named_parameters())

        loss = 0.
        for weight_name, (new_w, old_w) in dict_intersection(new_weights, old_weights):
            loss += torch.dist(new_w, old_w.type_as(new_w), p=self.options.distance_norm)

        self._i += 1
        ewc_loss = Loss(
            name=self.name,
            loss=loss,
        )
        return ewc_loss
