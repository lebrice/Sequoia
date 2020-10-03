from abc import ABC
from dataclasses import dataclass
from typing import ClassVar, Type, List

from simple_parsing import mutable_field

from common.loss import Loss
from settings import (Results, SettingType, ClassIncrementalSetting)
from utils.logging_utils import get_logger

from .method import Method
from .models import Model

from .models import Model
from .models.model_addons import ClassIncrementalModel as ClassIncrementalModelMixin
from common.callbacks import KnnCallback
from pytorch_lightning import Callback


logger = get_logger(__file__)

ClassIncrementalModel = Model

@dataclass
class ClassIncrementalMethod(Method, target_setting=ClassIncrementalSetting):
    """ Base class for methods that want to target a Task Incremental setting.

    TODO: Figure out if this makes sense, really.

    Method which is to be applied to a class incremental CL problem setting.
    """
    name: ClassVar[str] = "baseline"
    # HyperParameters of the LightningModule. Overwrite this in your class.
    hparams: Model.HParams = mutable_field(Model.HParams)
    # Options for the KNN classifier callback, which is used to evaluate the
    # quality of the representations on each test and val task after each
    # training epoch.
    knn_callback: KnnCallback = mutable_field(KnnCallback)

    def __post_init__(self):
        super().__post_init__()
        self.model: Model

    def create_callbacks(self, setting: SettingType) -> List[Callback]:
        from common.callbacks.vae_callback import SaveVaeSamplesCallback
        return super().create_callbacks(setting) + [
            self.knn_callback,
            SaveVaeSamplesCallback(),
        ]

    def train(self, setting: ClassIncrementalSetting) -> None:
        """ Trains the model. Overwrite this to customize training. """
        self.trainer.fit_setting(setting=setting, model=self.model)

    def model_class(self, setting: SettingType) -> Type[Model]:
        if issubclass(setting, ClassIncrementalSetting):
            return Model
        raise NotImplementedError(f"No model registered for setting {setting}!")

    def create_model(self, setting: ClassIncrementalSetting) -> Model:
        """Creates the Model (a LightningModule).

        The model should accept a Setting or datamodule in its constructor.

        Args:
            setting (Setting): The experimental setting.

        Returns:
            ClassIncrementalModel: The Model that is to be applied to that setting.
        """
        super().create_model
        return self.model_class(setting)(
            setting=setting,
            hparams=self.hparams,
            config=self.config,
        )


if __name__ == "__main__":
    ClassIncrementalMethod.main()
