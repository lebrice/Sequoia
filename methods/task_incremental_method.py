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

    # Options for the KNN classifier callback, which is used to evaluate the
    # quality of the representations on each test and val task after each
    # training epoch.
    knn_callback: KnnCallback = mutable_field(KnnCallback)

    def __post_init__(self):
        super().__post_init__()
        self.setting: TaskIncrementalSetting
        self.model: TaskIncrementalModel

    def train(self, setting: TaskIncrementalSetting) -> None:
        """ Trains the model. Overwrite this to customize training. """
        # Just a sanity check:
        assert self.model.setting is self.model.datamodule is setting
        n_tasks = setting.nb_tasks
        logger.info(f"Number of tasks: {n_tasks}")
        logger.info(f"Number of classes in task: {setting.num_classes}")

        for i in range(n_tasks):
            logger.info(f"Starting task #{i}")
            self.model.on_task_switch(i)
            assert self.model.setting.current_task_id == setting.current_task_id == i
            self.trainer.fit(self.model, datamodule=setting)
    
    @singledispatchmethod
    def model_class(self, setting: SettingType) -> Type[Model]:
        raise NotImplementedError(f"No model registered for setting {setting}!")
    
    @model_class.register
    def _(self, setting: TaskIncrementalSetting) -> Type[TaskIncrementalModel]:
        return TaskIncrementalModel

    def create_model(self, setting: TaskIncrementalSetting) -> TaskIncrementalModel:
        """Creates the Model (a LightningModule).

        The model should accept a Setting or datamodule in its constructor.

        Args:
            setting (Setting): The experimental setting.

        Returns:
            TaskIncrementalModel: The Model that is to be applied to that setting.
        """
        return self.model_class(setting)(
            setting=setting,
            hparams=self.hparams,
            config=self.config,
        )

    def task_switch(self, task_id: int) -> None:
        self.model.on_task_switch(task_id)


if __name__ == "__main__":
    TaskIncrementalMethod.main()
