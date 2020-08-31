from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from common.config import Config
from settings import TaskIncrementalSetting
from simple_parsing import mutable_field

from .class_incremental_model import ClassIncrementalModel
from .output_heads import OutputHead

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
        # TODO: It makes no sense to have multihead=True when the model doesn't
        # have access to task labels. Need to figure out how to manage this between TaskIncremental and Classifier.
        multihead: bool = False

    def __init__(self, setting: TaskIncrementalSetting, hparams: HParams, config: Config):
        if not isinstance(setting, TaskIncrementalSetting):
            raise RuntimeError(
                f"Can only apply this model on a {TaskIncrementalSetting} or "
                f"on a setting which inherits from it! "
                f"(given setting is of type {type(setting)})."
            )
        super().__init__(setting=setting, hparams=hparams, config=config)
        self.hp: TaskIncrementalModel.HParams
        self.setting: TaskIncrementalSetting

        if self.hp.multihead:
            # TODO: Actually implement something that uses this setting property
            # (task_label_is_readable), as it is not used anywhere atm really.
            # Maybe when we implement something like task-free CL? 
            assert self.setting.task_label_is_readable, (
                "Using a multihead model in a setting where the task label "
                "can't be read?"
            )
            self.output_heads: Dict[str, OutputHead] = nn.ModuleDict()
            self.output_heads[str(self.setting.current_task_id)] = self.create_output_head()
