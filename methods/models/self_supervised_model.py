""" Base class for a Self-Supervised model.

This is meant to be a kind of 'Mixin' that you can use and extend in order
to add self-supervised losses to your model.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, cast

from torch import Tensor

from common.config import Config
from common.loss import Loss
from common.tasks import AEReconstructionTask, VAEReconstructionTask, EWCTask
from common.tasks.auxiliary_task import AuxiliaryTask
from common.tasks.simclr import SimCLRTask
from settings import Setting, SettingType
from simple_parsing import mutable_field
from utils.logging_utils import get_logger
from utils.module_dict import ModuleDict

from .model import Model

logger = get_logger(__file__)
from typing import TypeVar
HParamsType = TypeVar("HparamsType", bound="SelfSupervisedModel.HParams")

class SelfSupervisedModel(Model[SettingType]):
    """
    Model 'mixin' that adds various configurable self-supervised losses.
    """
    @dataclass
    class HParams(Model.HParams):
        """Hyperparameters of a Self-Supervised method. """
        simclr: Optional[SimCLRTask.Options] = None
        vae: Optional[VAEReconstructionTask.Options] = None
        ae: Optional[AEReconstructionTask.Options] = None
        ewc: Optional[EWCTask.Options] = None

    def __init__(self, setting: Setting, hparams: HParams, config: Config):
        super().__init__(setting, hparams, config)
        self.hp: HParamsType
        # Dictionary of auxiliary tasks.
        self.tasks: Dict[str, AuxiliaryTask] = self.create_auxiliary_tasks()

        for task_name, task in self.tasks.items():
            logger.debug("Auxiliary tasks:")
            assert isinstance(task, AuxiliaryTask), f"Task {task} should be a subclass of {AuxiliaryTask}."
            if task.coefficient != 0:
                logger.debug(f"\t {task_name}: {task.coefficient}")
                logger.info(f"enabling the '{task_name}' auxiliary task (coefficient of {task.coefficient})")
                task.enable()

    def get_loss(self, x: Tensor, y: Tensor = None, loss_name: str = "") -> Loss:
        # TODO: Need to split up the get_loss function so it also kinda works
        # for active settings, (i.e. first part does the prediction, second part
        # for getting the loss).
        assert loss_name
        x, y = self.preprocess_batch(x, y)
        h_x = self.encode(x)
        y_pred = self.output_task(h_x)
        # Get the loss of the base model, to which we will add the
        # self-supervised loss signals.
        # NOTE: We pass the tensors into the Loss object so the metrics are
        total_loss = Loss(name=loss_name, x=x, h_x=h_x, y_pred=y_pred, y=y)
        if y is not None:
            supervised_loss = self.output_head.get_loss(x=x, h_x=h_x, y_pred=y_pred, y=y)
            total_loss += supervised_loss

        # Add the self-supervised losses from all the enabled auxiliary tasks.
        for task_name, aux_task in self.tasks.items():
            assert task_name, "Auxiliary tasks should have a name!"
            
            if aux_task.enabled:
                aux_loss: Loss = aux_task.get_loss(x=x, h_x=h_x, y_pred=y_pred, y=y)
                # Scale the loss by the corresponding coefficient.
                aux_loss *= aux_task.coefficient

                if self.config.verbose:
                    logger.debug(f"aux task {task_name}: Loss = {aux_loss}")
                total_loss += aux_loss

        if not self.config.debug:
            # Drop all the tensors, since we aren't debugging.
            total_loss.clear_tensors()

        return total_loss

    def create_auxiliary_tasks(self) -> Dict[str, AuxiliaryTask]:
        # Share the relevant parameters with all the auxiliary tasks.
        # We do this by setting class attributes.
        # TODO: Make sure that we aren't duplicating all of the model's weights
        # by setting a class attribute.
        AuxiliaryTask.model = self
        AuxiliaryTask.hidden_size = self.hidden_size
        AuxiliaryTask.input_shape = self.input_shape
        AuxiliaryTask.encoder = self.encoder
        AuxiliaryTask.classifier = self.output_head
        AuxiliaryTask.preprocessing = self.preprocess_batch

        tasks: Dict[str, AuxiliaryTask] = ModuleDict()

        # TODO(@lebrice): Should we create the tasks even if they aren't used,
        # and then 'enable' them when they are needed? (I'm thinking that maybe
        # being enable/disable auxiliary tasks when needed might be useful
        # later?)
        if self.hp.simclr and self.hp.simclr.coefficient:
            tasks[SimCLRTask.name] = SimCLRTask(options=self.hp.simclr)
        if self.hp.vae and self.hp.vae.coefficient:
            tasks[VAEReconstructionTask.name] = VAEReconstructionTask(options=self.hp.vae)
        if self.hp.ae and self.hp.ae.coefficient:
            tasks[AEReconstructionTask.name] = AEReconstructionTask(options=self.hp.ae)
        if self.hp.ewc and self.hp.ewc.coefficient:
            tasks[EWCTask.name] = EWCTask(options=self.hp.ewc)

        return tasks
    
    def on_task_switch(self, task_id: int) -> None:
        """Called when switching between tasks.

        Args:
            task_id (int): the Id of the task.
        """
        for task_name, task in self.tasks.items():
            if task.enabled:
                task.on_task_switch(task_id=task_id)
        super().on_task_switch(task_id=task_id)
