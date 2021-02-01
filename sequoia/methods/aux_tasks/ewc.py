"""Elastic Weight Consolidation as an Auxiliary Task.

This is a simplified version of EWC, that only currently uses the L2 norm, rather
than the Fisher Information Matrix.

TODO: If it's worth it, we could re-add the 'real' EWC using the nngeometry
package, (which I don't think we need to have as a submodule).
"""

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from typing import (Callable, ClassVar, Deque, Dict, Iterator, List, Mapping,
                    MutableMapping, Optional, Tuple, Type, TypeVar, Union)

import torch
from gym.spaces.utils import flatdim
from nngeometry.generator.jacobian import Jacobian
from nngeometry.layercollection import LayerCollection
from nngeometry.metrics import FIM
from nngeometry.object.pspace import PMatAbstract, PMatDiag, PMatKFAC, PVector
from simple_parsing import choice
from torch import Tensor, nn
from torch.autograd import Variable
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader, Dataset, TensorDataset

from sequoia.common.loss import Loss
from sequoia.methods.aux_tasks.auxiliary_task import AuxiliaryTask
from sequoia.methods.models.forward_pass import ForwardPass
from sequoia.methods.models.output_heads import (ClassificationHead,
                                                 RegressionHead)
from sequoia.settings.base.objects import Observations
from sequoia.utils.logging_utils import get_logger
from sequoia.utils.utils import dict_intersection

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

        # Batchsize to be used when computing FIM
        batch_size_fim: int = 64
        # Number of observations to use for FIM calculation
        sample_size_fim: int = 400
        # Fisher information type  (diagonal or block diagobnal)
        fim_representation: Type[PMatAbstract] = choice(
            {"diagonal": PMatDiag, "block_diagonal": PMatKFAC,}, default=PMatKFAC,
        )

    def __init__(
        self,
        *args,
        name: str = None,
        options: "EWC.Options" = None,
        method: str = "generic",
        **kwargs,
    ):
        super().__init__(*args, options=options, name=name, **kwargs)
        self.options: EWCTask.Options
        self.previous_task: int = None
        self._i: int = 0
        self.n_switches: int = 0
        self.method = method if method in ["dqn", "a2c"] else "generic"
        logger.info(f"EWC will be applied in {self.method} mode.")

        self.previous_model_weights: Optional[PVector] = None
        self.observation_collector: Deque[Observations] = deque(
            maxlen=self.options.sample_size_fim
        )
        self.fisher_information_matrices: List[PMatAbstract] = []

    def consolidate(self, new_fims: List[PMatAbstract], task: int) -> None:
        if not self.fisher_information_matrices:
            self.fisher_information_matrices = new_fims
            return

        for i, (fim_previous, fim_new) in enumerate(zip(self.fisher_information_matrices, new_fims)):
            if fim_previous is None:
                self.fisher_information_matrices[i] = fim_new
            else:
                # consolidate the FIMs
                # consolidate the fim_new into fim_previous in place
                if isinstance(fim_new, PMatDiag):
                    # TODO: This is some kind of weird online-EWC related magic:
                    fim_previous.data = (
                        deepcopy(fim_new.data) + fim_previous.data * (task)
                    ) / (task + 1)

                elif isinstance(fim_new.data, dict):
                    # TODO: This is some kind of weird online-EWC related magic:
                    for key, (prev_param, new_param) in dict_intersection(fim_previous.data, fim_new.data):
                        for prev_item, new_item in zip(prev_param, new_param):
                            prev_item.data = (prev_item.data * task + deepcopy(new_item.data)) / (task + 1)

                self.fisher_information_matrices[i] = fim_previous

    def on_task_switch(self, task_id: Optional[int]):
        """ Executed when the task switches (to either a known or unknown task).
        """
        if not self.enabled:
            return

        logger.info(f"On task switch called: task_id={task_id}")
        # if task_id != 0:
        #     # BUG: IDK why, but the encoder is now on the CPU rather than on the device
        #     # it was on previously?
        #     shared_net_device = list(self._model.encoder.parameters())[0].device
        #     for fim in self.fisher_information_matrices:
        #         fim.data = {
        #             k: (param[0].to(shared_net_device, param[1].to(shared_net_device))) for k, param in fim.data.items()
        #         } 

        if self._shared_net is None:
            logger.info(
                f"On task switch called: task_id={task_id}, EWC can not be applied as the used net has no shared part."
            )

        # TODO: deal with situation when task_id is None
        elif self.previous_task is None and self.n_switches == 0 and not task_id:
            self.previous_task = task_id
            logger.info("Starting the first task, no EWC update.")
            self.n_switches += 1

        elif task_id is None or task_id > self.previous_task and self._model.training:
            # we dont want to go here at test time.
            # NOTE: We also switch between unknown tasks.
            logger.info(
                f"Switching tasks: {self.previous_task} -> {task_id}: "
                f"Updating the EWC 'anchor' weights."
            )
            self.previous_task = task_id
            device = self._model.config.device
            self.previous_model_weights = PVector.from_model(self._shared_net.to(device)).clone().detach()

            # Create a Dataloader from the stored observations.
            obs_type: Type[Observations] = type(self.observation_collector[0])
            dataset = [obs.as_namedtuple() for obs in self.observation_collector]
            # Or, alternatively:
            # stacked_observations: Observations = obs_type.stack(self.observation_collector)
            # dataset = TensorDataset(*stacked_observations.as_namedtuple())

            dataloader = DataLoader(dataset, batch_size=None, collate_fn=None)
            
            variant: str
            if isinstance(self._model.output_head, ClassificationHead):
                variant = "classif_logits"
                n_output = self._model.action_space.n

                def fim_function(*inputs) -> Tensor:
                    observations = obs_type(*inputs).to(self._model.device)
                    forward_pass: ForwardPass = self._model(observations)
                    actions = forward_pass.actions
                    return actions.logits

            elif isinstance(self._model.output_head, RegressionHead):
                variant = "regression"
                n_output = flatdim(self._model.action_space)

                def fim_function(*inputs) -> Tensor:
                    observations = obs_type(*inputs).to(self._model.device)
                    forward_pass: ForwardPass = self._model(observations)
                    actions = forward_pass.actions
                    return actions.y_pred
            else:
                raise NotImplementedError("TODO")
            
            new_fim = FIM(
                model=self._shared_net,
                loader=dataloader,
                representation=self.options.fim_representation,
                n_output=n_output,
                variant=variant,
                function=fim_function,
                device=self._model.device,
            )

            # TODO: There was maybe an idea to use another fisher information matrix for
            # the critic in A2C, but not doing that atm.
            new_fims = [new_fim]

            self.consolidate(new_fims, task=self.previous_task)
            self.n_switches += 1
            self.observation_collector.clear()

    @property
    def _shared_net(self) -> Optional[nn.Module]:
        """
        Returns 'None' if there is not shared network part, othervise returns the shared net
        """
        if self._model.encoder is None:
            return None
        elif isinstance(self._model.encoder, nn.Sequential):
            if len(self._model.encoder) == 0:
                return None
        return self._model.encoder

    def get_loss(self, forward_pass: ForwardPass, y: Tensor = None) -> Loss:
        """ Gets the EWC loss.
        """
        if self._model.training:
            self.observation_collector.append(forward_pass.observations)

        if self.previous_task is None or not self.enabled or self._shared_net is None:
            # We're in the first task: do nothing.
            return Loss(name=self.name)

        loss = 0.0
        v_current = PVector.from_model(self._shared_net)
        
        for fim in self.fisher_information_matrices:
            diff = v_current - self.previous_model_weights
            loss += fim.vTMv(diff)

        if loss != 0.:
            # FIXME: Take this out, just debugging for now.
            assert False, loss

        self._i += 1
        ewc_loss = Loss(name=self.name, loss=loss)
        return ewc_loss
