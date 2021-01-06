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


logger = get_logger(__file__)


class EWCTask(AuxiliaryTask):
    """ Elastic Weight Consolidation, implemented as a 'self-supervision-style' 
    Auxiliary Task.
    """
    name: str = "ewc"

    @dataclass    
    class Options(AuxiliaryTask.Options):
        """ Options of the EWC auxiliary task. """

        #Batchsize to be used when computing FIM
        batch_size_fim: int = 64
        # Number of observations to use for FIM calculation
        total_steps_fim: int = 1000
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
        self.observation_collector = deque(maxlen=self.options.total_steps_fim)

    def consolidate(self, new_fims:List[PMatAbstract], task:int) -> None:
        """
        Consolidates the previous FIMs and the new onces.
        See online EWC in https://arxiv.org/pdf/1805.06370.pdf.
        """
        if self.FIMs is None:
            self.FIMs = new_fims
            return 
        assert len(new_fims)==len(self.FIMs)
        for i, (fim_previous, fim_new) in enumerate(zip(self.FIMs, new_fims)):
            if fim_previous is None:
                self.FIMs[i] = fim_new
            else:
                #consolidate the FIMs
                self.FIMs[i] = EWCPolicy._consolidate_fims(fim_previous,fim_new, task)

    @staticmethod
    def _consolidate_fims(fim_previous: PMatAbstract, fim_new: PMatAbstract, task:int) -> PMatAbstract:
        #consolidate the fim_new into fim_previous in place
        if isinstance(fim_new, PMatDiag):  
            fim_previous.data = ((deepcopy(fim_new.data)) + fim_previous.data * (task)) / (task + 1)

        elif isinstance(fim_new.data, dict):  
            for (n, p), (n_, p_) in zip(fim_previous.data.items(),fim_new.data.items()):
                for item, item_ in zip(p, p_):
                    item.data = ((item.data*(task))+deepcopy(item_.data))/(task+1)
        return fim_previous
        
    def on_task_switch(self, task_id: Optional[int]): #, dataloader: DataLoader, method: str = 'a2c')-> None:
        """ Executed when the task switches (to either a known or unknown task).
        """
        if not self.enabled:
            return
        logger.info(f"On task switch called: task_id={task_id}")

        #TODO: deal with situation hen task_id is None
        if self.previous_task is None and self.n_switches == 0 and not task_id:
            self.previous_task = task_id
            logger.info("Starting the first task, no EWC update.")
            self.n_switches += 1
        
        elif task_id is None or task_id > self.previous_task and self._model.training:
            #we dont want to go here at test tiem
            # NOTE: We also switch between unknown tasks.
            logger.info(f"Switching tasks: {self.previous_task} -> {task_id}: "
                         f"Updating the EWC 'anchor' weights.")
            self.previous_task = task_id
            self.previous_model_weights = PVector.from_model(self._model).clone().detach()

            def splitbatch(x: List[Observations]) -> List[Observations]:
                """ Splits a list of batched observatins into a list of observations with batch size 1
                """
                res = []
                for obs in x:
                    res2 = []
                    for i in range(len(obs)):
                        res2.append([obs[:,i][f] for f in obs.field_names])
                    res +=res2                    
                return res
            #observation_collection = list(map(splitbatch, self.observation_collector))
            dataloader = DataLoader(splitbatch(self.observation_collector)) #self._model.setting.train_env #DataLoader(TensorDataset(observation_collection), batch_size=self.options.batch_size_fim, shuffle=False)
            
            new_fims=[]
            if self.method=='dqn':
                function=self._model.q_net
                n_output=self.action_space.n
                method = 'dqn'
            elif self.method=='a2c':
                function=self._model
                n_output=1
                method = 'a2c'
            else:
                function=lambda *x: self._model(self.observation_collector[0].__class__(*x)).logits
                n_output=self._model.action_space.n
                method = 'classifimlogits'

            new_fim = FIM(model=self._model,
                        loader=dataloader,   
                        representation=self.FIM_representation,
                        n_output=n_output,
                        variant=method,
                        function = function,
                        device=self.device.type)
            new_fims.append(new_fim)
            if self.method=='a2c':
                #apply EWC also to the value net
                new_fim_critic = FIM(model=self,    
                        loader=dataloader,   
                        representation=self.FIM_representation,
                        n_output=1,
                        variant='regression',
                        function = lambda *x: self(x[0])[1],
                        device=self.device.type)
                new_fims.append(new_fim_critic)
            self.consolidate(new_fims,task=self.previous_task)
            self.n_switches += 1
            self.observation_collector = deque(maxlen=self.options.total_steps_fim)
    
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
        if self.previous_task is None or not self.enabled or self.FIMs is None:
            # We're in the first task: do nothing.
            return Loss(name=self.name)
        
        loss = 0. 
        v_current = PVector.from_model(self._model)
        for fim in self.FIMs:
            loss += fim.vTMv(v_current - self.previous_model_weights)

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
