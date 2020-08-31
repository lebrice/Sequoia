from abc import ABC
from dataclasses import dataclass
from typing import ClassVar, Type, List

from simple_parsing import mutable_field
from singledispatchmethod import singledispatchmethod

from common.loss import Loss
from settings import (Results, SettingType, TaskIncrementalSetting)
from utils.logging_utils import get_logger

from .models import Model
from .class_incremental_method import ClassIncrementalMethod
from .models.task_incremental_model import TaskIncrementalModel
from common.callbacks import KnnCallback
from pytorch_lightning import Callback


logger = get_logger(__file__)


@dataclass
class TaskIncrementalMethod(ClassIncrementalMethod, target_setting=TaskIncrementalSetting):
    """ Base class for methods that want to target a Task Incremental setting.
    """
    # HyperParameters of the LightningModule. Overwrite this in your class.
    hparams: TaskIncrementalModel.HParams = mutable_field(TaskIncrementalModel.HParams)

    def __post_init__(self):
        super().__post_init__()
        self.setting: TaskIncrementalSetting
        self.model: TaskIncrementalModel

    @singledispatchmethod
    def model_class(self, setting: SettingType) -> Type[Model]:
        raise NotImplementedError(f"No model registered for setting {setting}!")
    
    @model_class.register
    def _(self, setting: TaskIncrementalSetting) -> Type[TaskIncrementalModel]:
        return TaskIncrementalModel

    def task_switch(self, task_id: int) -> None:
        logger.info(f"task_switch called on the TaskIncrementalMethod (task_id={task_id})")
        self.model.on_task_switch(task_id, training=False)


if __name__ == "__main__":
    TaskIncrementalMethod.main()
