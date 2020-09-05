from dataclasses import dataclass
from typing import Type

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

    TODO: This isn't used anywhere atm, but the idea is that it would save a
    little bit of boilerplate code.

    This is essentially just the same as ClassIncrementalMethod, but uses the
    TaskIncrementalModel (instead of ClassIncrementalModel), by not having to
    subclass ClassIncremntalMethod and return a TaskIncrementalModel in
    `model_class()`.
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

if __name__ == "__main__":
    TaskIncrementalMethod.main()
