""" Example of an auxiliary task. This is basically the same as in the examples. """

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch import Tensor

from sequoia.common import Loss
from sequoia.utils.utils import dict_intersection
from sequoia.utils.logging_utils import get_logger

from .auxiliary_task import AuxiliaryTask

logger = get_logger(__file__)


class ExampleAuxTask(AuxiliaryTask):
    """ "EWC-like" regularization using simple L2 distance, implemented as a
    'self-supervision-style' Auxiliary Task.
    """

    name: str = "l2_cl_regularization"

    @dataclass
    class Options(AuxiliaryTask.Options):
        """Options of the EWC auxiliary task."""

        # Wether to use the absolute difference of the weights or the difference
        # in the `regularize` method below.
        use_abs_diff: bool = False
        # The norm term for the 'distance' between the current and old weights.
        distance_norm: int = 2

    def __init__(self, *args, name: str = None, options: "EWC.Options" = None, **kwargs):
        super().__init__(*args, options=options, name=name, **kwargs)
        self.options: ExampleAuxTask.Options

        self.previous_task: int = None
        # TODO: Figure out a clean way to persist this dict into the state_dict.
        self.previous_model_weights: Dict[str, Tensor] = {}
        self._i: int = 0
        self.n_switches: int = 0

    def state_dict(self, *args, **kwargs) -> Dict:
        state = super().state_dict(*args, **kwargs)
        state.update(self.previous_model_weights)
        return state

    def load_state_dict(
        self, state_dict: Dict[str, Tensor], strict: bool = True
    ) -> Tuple[List[str], List[str]]:
        missing: List[str]
        unexpected: List[str]
        missing, unexpected = super().load_state_dict(state_dict=state_dict, strict=False)
        if unexpected and not self.previous_model_weights:
            # Create the previous model weights, if needed.
            self.previous_model_weights.update(
                deepcopy({k: v.detach() for k, v in self.model.named_parameters()})
            )
        # TODO: Make sure that the model itself (i.e. its output heads, etc) gets
        # restored before this here.
        for key in unexpected.copy():
            if key in self.previous_model_weights:
                # Update the value in the 'previous model weights' dict.
                self.previous_model_weights[key] = state_dict[key]
                unexpected.remove(key)
        return missing, unexpected

    def disable(self):
        """Disable the EWC loss."""
        # save a little bit of memory by clearing the weights.
        self.previous_model_weights.clear()
        return super().disable()

    def enable(self):
        # old_weights = parameters_to_vector(self.model.parameters())
        # self.register_buffer("old_weights", old_weights, persistent = True)
        return super().enable()

    def on_task_switch(self, task_id: int) -> None:
        """Executed when the task switches (to either a new or known task)."""
        if not self.enabled:
            return
        if self.previous_task is None and self.n_switches == 0:
            logger.debug(f"Starting the first task, no EWC update.")
        elif task_id is None or task_id != self.previous_task:
            logger.debug(
                f"Switching tasks: {self.previous_task} -> {task_id}: "
                f"Updating the EWC 'anchor' weights."
            )
            self.previous_task = task_id
            self.previous_model_weights.clear()
            self.previous_model_weights.update(
                deepcopy({k: v.detach() for k, v in self.model.named_parameters()})
            )
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

        loss = 0.0
        for weight_name, (new_w, old_w) in dict_intersection(new_weights, old_weights):
            loss += torch.dist(new_w, old_w.type_as(new_w), p=self.options.distance_norm)

        self._i += 1
        ewc_loss = Loss(
            name=self.name,
            loss=loss,
        )
        return ewc_loss
