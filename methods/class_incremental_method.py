from abc import ABC
from dataclasses import dataclass
from typing import ClassVar, Type, List

from simple_parsing import mutable_field
from singledispatchmethod import singledispatchmethod

from common.loss import Loss
from settings import (Results, SettingType, ClassIncrementalSetting)
from utils.logging_utils import get_logger

from .method import Method
from .models import Model
from .models.class_incremental_model import ClassIncrementalModel
from common.callbacks import KnnCallback
from pytorch_lightning import Callback


logger = get_logger(__file__)

@dataclass
class ClassIncrementalMethod(Method, target_setting=ClassIncrementalSetting):
    """ Base class for methods that want to target a Task Incremental setting.

    TODO: Figure out if this makes sense, really.

    Method which is to be applied to a class incremental CL problem setting.
    """
    name: ClassVar[str] = ""
    # HyperParameters of the LightningModule. Overwrite this in your class.
    hparams: ClassIncrementalModel.HParams = mutable_field(ClassIncrementalModel.HParams)

    # Options for the KNN classifier callback, which is used to evaluate the
    # quality of the representations on each test and val task after each
    # training epoch.
    knn_callback: KnnCallback = mutable_field(KnnCallback)

    def __post_init__(self):
        super().__post_init__()
        self.setting: ClassIncrementalSetting
        self.model: ClassIncrementalModel

    def create_callbacks(self, setting: SettingType) -> List[Callback]:
        return super().create_callbacks(setting) + [
            self.knn_callback,
        ]

    def train(self, setting: ClassIncrementalSetting) -> None:
        """ Trains the model. Overwrite this to customize training. """
        # Just a sanity check:
        assert self.model.setting is setting
        n_tasks = setting.nb_tasks
        logger.info(f"Number of tasks: {n_tasks}")
        logger.info(f"Number of classes in task: {setting.num_classes}")

        for i in range(n_tasks):
            logger.info(f"Starting task #{i}")
            if setting.task_labels_at_train_time:
                # This is always true in the ClassIncremental & TaskIncremental
                # settings for now.
                self.model.on_task_switch(i)
            # TODO: Make sure the Trainer really does max_epochs per task, and
            # not max_epochs on the first task and 0 on the others.
            assert self.model.setting.current_task_id == setting.current_task_id == i
            self.trainer.fit(self.model, datamodule=setting)

    @singledispatchmethod
    def model_class(self, setting: SettingType) -> Type[Model]:
        raise NotImplementedError(f"No model registered for setting {setting}!")

    @model_class.register
    def _(self, setting: ClassIncrementalSetting) -> Type[ClassIncrementalModel]:
        return ClassIncrementalModel

    def create_model(self, setting: ClassIncrementalSetting) -> ClassIncrementalModel:
        """Creates the Model (a LightningModule).

        The model should accept a Setting or datamodule in its constructor.

        Args:
            setting (Setting): The experimental setting.

        Returns:
            ClassIncrementalModel: The Model that is to be applied to that setting.
        """
        return self.model_class(setting)(
            setting=setting,
            hparams=self.hparams,
            config=self.config,
        )

    def on_task_switch(self, task_id: int) -> None:
        self.model.on_task_switch(task_id)


if __name__ == "__main__":
    ClassIncrementalMethod.main()
