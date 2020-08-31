from simple_parsing.helpers import FlattenedAccess
from dataclasses import dataclass
from typing import Dict, List, Type

from pytorch_lightning import Callback
from simple_parsing import mutable_field
from singledispatchmethod import singledispatchmethod
from torch import Tensor

from common.config import Config
from common.loss import Loss
from common.tasks import AuxiliaryTask
from settings import ClassIncrementalSetting
from settings.base.setting import Setting, SettingType
from utils import get_logger

from .method import Method
from .models import HParams, Model
from .models.self_supervised_model import SelfSupervisedModel
from .class_incremental_method import (ClassIncrementalMethod,
                                       ClassIncrementalModel)

logger = get_logger(__file__)
from common.tasks.simclr import SimCLRTask


@dataclass
class SelfSupervisedMethod(ClassIncrementalMethod):
    """ Method where self-supervised learning is used to learn representations.

    The representations of the model are learned either jointly with the
    downstream task (e.g. classification) loss, or only through self-supervision
    when `detach_output_head` is set to True.

    IDEA: maybe we could instead make the 'method' an optional mixin, and create
    the "real" method class dynamically?
    """
    # Hyperparameters of the model.
    # TODO: If we were to support more models, we might have a problem trying to
    # get the help text of each type of hyperparameter to show up. We can still
    # parse them just fine by calling .from_args() on them, but still, would be
    # better if the help text were visible from the command-line.
    hparams: ClassIncrementalModel.HParams = mutable_field(ClassIncrementalModel.HParams)

    def create_callbacks(self, setting: SettingType) -> List[Callback]:
        from common.callbacks.vae_callback import SaveVaeSamplesCallback
        return super().create_callbacks(setting) + [
            SaveVaeSamplesCallback(),
        ]


if __name__ == "__main__":
    SelfSupervisedMethod.main()
