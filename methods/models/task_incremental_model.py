from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from common.config import Config
from settings import TaskIncrementalSetting

from .class_incremental_model import ClassIncrementalModel

SettingType = TypeVar("SettingType", bound=TaskIncrementalSetting)


class TaskIncrementalModel(ClassIncrementalModel[SettingType]):
    """ Extension of the Classifier LightningModule aimed at CL settings.
    TODO: Add the stuff related to multihead/continual learning here?
    """

    @dataclass
    class HParams(ClassIncrementalModel.HParams):
        """ Hyperparameters specific to a Continual Learning classifier.
        TODO: Add any hyperparameters specific to CL here.
        """
        # Wether to create one output head per task.
        # TODO: Will this ever be False? I'm pretty sure we're saying that task
        # labels always given in Task Incremental, so does it ever make sense
        # to use a single-head Model when the task labels are always given?
        multihead: bool = True

    def __init__(self, setting: SettingType, hparams: HParams, config: Config):
        super().__init__(setting=setting, hparams=hparams, config=config)
        self.hp: TaskIncrementalModel.HParams
        self.setting: SettingType
