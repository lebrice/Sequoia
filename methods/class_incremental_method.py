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
    """ Base class for methods that want to specifically target the ClassIncremental setting.

    TODO: This basically only adds a KNN Callback and VAE Callback
    """
    # HyperParameters of the LightningModule. Overwrite this in your class.
    hparams: Model.HParams = mutable_field(Model.HParams)
    # Adds Options for the KNN classifier callback, which is used to evaluate
    # the quality of the representations on each test and val task after each
    # training epoch.
    # TODO: Debug/test this callback to make sure it still works fine.
    knn_callback: KnnCallback = mutable_field(KnnCallback)

    def __post_init__(self):
        super().__post_init__()
        self.model: Model

    def create_callbacks(self, setting: SettingType) -> List[Callback]:
        # TODO: Move this to something like a `configure_callbacks` method 
        # in the corresponding models, since PL might add it.
        from common.callbacks.vae_callback import SaveVaeSamplesCallback
        return super().create_callbacks(setting) + [
            self.knn_callback,
            SaveVaeSamplesCallback(),
        ]

if __name__ == "__main__":
    ClassIncrementalMethod.main()
