""" Base class for a Self-Supervised model.

This is meant to be a kind of 'Mixin' that you can use and extend in order
to add self-supervised losses to your model.
"""

import warnings
from dataclasses import dataclass
from typing import Dict, Optional, TypeVar

from torch import Tensor, nn

from sequoia.common.config import Config
from sequoia.common.loss import Loss
from sequoia.methods.aux_tasks.auxiliary_task import AuxiliaryTask
from sequoia.settings import Rewards, Setting, SettingType
from sequoia.utils.logging_utils import get_logger
from sequoia.utils.utils import flatten_dict
from .base_model import BaseModel

# from sequoia.utils.module_dict import ModuleDict


logger = get_logger(__file__)
HParamsType = TypeVar("HParamsType", bound="SelfSupervisedModel.HParams")


class SelfSupervisedModel(BaseModel[SettingType]):
    """
    Model 'mixin' that adds support for modular, configurable "auxiliary tasks".
    
    These auxiliary tasks are used to get a self-supervised loss to train on
    when labels aren't available.
    """

    @dataclass
    class HParams(BaseModel.HParams):
        """Hyperparameters of a Self-Supervised method. """

        # simclr: Optional[SimCLRTask.Options] = None
        # vae: Optional[VAEReconstructionTask.Options] = None
        # ae: Optional[AEReconstructionTask.Options] = None

    def __init__(self, setting: Setting, hparams: HParams, config: Config):
        super().__init__(setting, hparams, config)
        self.hp: SelfSupervisedModel.HParams
        # Dictionary of auxiliary tasks.
        self.tasks: Dict[str, AuxiliaryTask] = nn.ModuleDict()

    def get_loss(
        self,
        forward_pass: Dict[str, Tensor],
        rewards: Rewards = None,
        loss_name: str = "",
    ) -> Loss:
        # Get the output task loss (the loss of the base model)
        loss: Loss = super().get_loss(
            forward_pass, rewards=rewards, loss_name=loss_name
        )

        # Add the self-supervised losses from all the enabled auxiliary tasks.
        for task_name, aux_task in self.tasks.items():
            assert task_name, "Auxiliary tasks should have a name!"
            if aux_task.enabled:
                # TODO: Auxiliary tasks all share the same 'y' for now, but it
                # might make more sense to organize this differently.
                y = rewards.y if rewards else None
                aux_loss: Loss = aux_task.get_loss(forward_pass, y=y)
                # Scale the loss by the corresponding coefficient before adding
                # it to the total loss.
                loss += aux_task.coefficient * aux_loss.to(self.device)
                if self.config.debug and self.config.verbose:
                    logger.debug(f"{task_name} loss: {aux_loss.total_loss}")

        return loss

    def add_auxiliary_task(
        self, aux_task: AuxiliaryTask, key: str = None, coefficient: float = None
    ) -> None:
        """ Adds an auxiliary task to the self-supervised model. """
        key = aux_task.name if key is None else key
        if key in self.tasks:
            raise RuntimeError(
                f"There is already an auxiliary task with name {key} in the model!"
            )
        self.tasks[key] = aux_task.to(self.device)
        if coefficient is not None:
            aux_task.coefficient = coefficient
        elif not aux_task.coefficient:
            warnings.warn(
                UserWarning(
                    f"Adding auxiliary task with name {key}, but with coefficient of 0.!"
                )
            )

        if aux_task.coefficient:
            aux_task.enable()

    def create_auxiliary_tasks(self) -> Dict[str, AuxiliaryTask]:
        # Share the relevant parameters with all the auxiliary tasks.
        # We do this by setting class attributes.
        # TODO: Make sure that we aren't duplicating all of the model's weights
        # by setting a class attribute.
        AuxiliaryTask._model = self
        AuxiliaryTask.hidden_size = self.hidden_size
        AuxiliaryTask.input_shape = self.input_shape
        AuxiliaryTask.encoder = self.encoder
        AuxiliaryTask.output_head = self.output_head
        # AuxiliaryTask.preprocessing = self.preprocess_batch

        tasks: Dict[str, AuxiliaryTask] = nn.ModuleDict()
        # TODO(@lebrice): Should we create the tasks even if they aren't used,
        # and then 'enable' them when they are needed? (I'm thinking that maybe
        # being enable/disable auxiliary tasks when needed might be useful
        # later?)
        # if self.hp.simclr and self.hp.simclr.coefficient:
        #     tasks[SimCLRTask.name] = SimCLRTask(options=self.hp.simclr)
        # if self.hp.vae and self.hp.vae.coefficient:
        #     tasks[VAEReconstructionTask.name] = VAEReconstructionTask(options=self.hp.vae)
        # if self.hp.ae and self.hp.ae.coefficient:
        #     tasks[AEReconstructionTask.name] = AEReconstructionTask(options=self.hp.ae)
        # if self.hp.ewc and self.hp.ewc.coefficient:
        #     tasks[EWCTask.name] = EWCTask(options=self.hp.ewc)

        return tasks

    def on_task_switch(self, task_id: Optional[int]) -> None:
        """Called when switching between tasks.

        Args:
            task_id (int): the Id of the task.
        """
        for task_name, task in self.tasks.items():
            if task.enabled:
                task.on_task_switch(task_id=task_id)
        super().on_task_switch(task_id=task_id)

    def shared_modules(self) -> Dict[str, nn.Module]:
        """Returns any trainable modules in `self` that are shared across tasks.

        By giving this information, these weights can then be used in
        regularization-based auxiliary tasks like EWC, for example.

        For the base model, this returns a dictionary with the encoder, for example.
        When using auxiliaryt tasks, they also add their shared weights, if any. 

        Returns
        -------
        Dict[str, nn.Module]:
            Dictionary mapping from name to the shared modules, if any.
        """
        shared_modules = super().shared_modules()
        for task_name, task in self.tasks.items():
            # TODO: What separator to use when dealing with nested dictionaries? I seem
            # to recall that ModuleDicts don't like some separators.
            sep = "."
            task_modules = task.shared_modules()
            flattened_task_modules = flatten_dict(task_modules, separator=sep)
            for module_name, module in flattened_task_modules.items():
                shared_modules[f"{task_name}{sep}{module_name}"] = module
        return shared_modules
