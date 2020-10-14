"""Elastic Weight Consolidation as an Auxiliary Task.

TODO: Refactor / Validate / test the EWC Auxiliary Task with the new setup.

"""
from copy import deepcopy
from dataclasses import dataclass
from typing import (Dict, Iterable, Iterator, List, Mapping, MutableMapping,
                    Optional, Tuple)

import torch
from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader

# from common.dict_buffer import DictBuffer
from common.loss import Loss
from common.task import Task
from common.tasks.auxiliary_task import AuxiliaryTask
from methods.models.output_heads import OutputHead
from utils import dict_intersection, dict_union
from utils.logging_utils import get_logger

logger = get_logger(__file__)


class EWCTask(AuxiliaryTask):
    """ Elastic Weight Consolidation, implemented as a 'self-supervision-style' 
    Auxiliary Task.
    """
    name: str = "ewc"

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
                 name: str = "ewc",
                 options: "EWC.Options" = None,
                 **kwargs):
        super().__init__(*args, name=name, options=options, **kwargs)
        self.name = name or type(self.name)
        self.options: EWCTask.Options
        self.previous_task: int = None
        # TODO: Figure out a clean way to persist this dict into the state_dict.
        self.previous_model_weights: Dict[str, Tensor] = {}
        self._i: int = 0
        self.n_switches: int = 0

    def state_dict(self, *args, **kwargs) -> Dict:
        state = super().state_dict(*args, **kwargs)
        for k, v in self.previous_model_weights.items():
            state[k] = v
        return state

    def load_state_dict(self, state_dict: Dict[str, Tensor], strict: bool = True):
        missing, unexpected = super().load_state_dict(state_dict=state_dict, strict=False)
        for k, v in self.previous_model_weights.items():
            if k in unexpected:
                self.previous_model_weights[k] = state_dict[k]

    def disable(self):
        # save a little bit of memory by clearing the weights.
        self.previous_model_weights.clear()
        # TODO: Should we also reset the 'previous_task' ?
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
        if self.previous_task is None and task_id is not None and self.n_switches == 0:
            logger.debug(f"Starting the first task, no EWC update.")
            pass
        elif task_id != self.previous_task:
            logger.debug(f"Switching tasks: {self.previous_task} -> {task_id}: "
                         f"Updating the EWC 'anchor' weights.")
            self.previous_task = task_id
            self.previous_model_weights.clear()
            self.previous_model_weights.update(deepcopy({
                k: v.detach() for k, v in self.model.named_parameters()
            }))
            # self.previous_model_weights.requires_grad_(False)
            # self.old_weights = parameters_to_vector(self.model.parameters())
        self.n_switches += 1

    def get_loss(self, forward_pass: Dict[str, Tensor], y: Tensor = None) -> Loss:
        """Gets the 'EWC' loss. 

        NOTE: This is a simplified version of EWC where the loss is the L2-norm
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
        for k, (new_w, old_w) in dict_intersection(new_weights, old_weights):
            # assert new_w.requires_grad
            # TODO: Does the ordering matter, for L1, for example?
            loss += torch.dist(new_w, old_w.type_as(new_w))

        self._i += 1
        ewc_loss = Loss(
            name=self.name,
            loss=loss,
        )
        return ewc_loss
