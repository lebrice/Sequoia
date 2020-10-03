"""TODO: Remove this. Basically only used to set the 'multihead' value to a
default of True rather than False, as it was set in its parent.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from common.config import Config
from settings import TaskIncrementalSetting
from simple_parsing import field

from .model import Model

SettingType = TypeVar("SettingType", bound=TaskIncrementalSetting)

# NOTE: The `Model` class uses a mixin to give it 'Class-Incremental' support.

class TaskIncrementalModel(Model[SettingType]):
    """ Extension of the Classifier LightningModule aimed at CL settings.
    TODO: Add the stuff related to multihead/continual learning here?
    """

    @dataclass
    class HParams(Model.HParams):
        """ Hyperparameters specific to a Continual Learning classifier.
        TODO: Add any hyperparameters specific to CL here.
        """
        # Wether to create one output head per task.
        # TODO: Will this ever be False? I'm pretty sure we're saying that task
        # labels are always given in Task Incremental, so does it ever make
        # sense to use a single-head Model when the task labels are always
        # given?
        # TODO: In other models, this has a default value of False, which works
        # fine with no argument passed, i.e. '--multihead' -> multihead=True.
        # However, in this particular model, the default value is set to `True`,
        # so if you used the argument without without a value, you'd get the
        # opposite of the default, in this case, False:
        # '--multihead' --> multihead=False.
        # This is slightly confusing, hence we're requiring a value here.
        multihead: bool = field(default=True, nargs=1)

        def __post_init__(self):
            super().__post_init__()
            if isinstance(self.multihead, list):
                assert len(self.multihead) == 1
                self.multihead = self.multihead[0]

    def __init__(self, setting: SettingType, hparams: HParams, config: Config):
        super().__init__(setting=setting, hparams=hparams, config=config)
        self.hp: TaskIncrementalModel.HParams
        self.setting: SettingType
