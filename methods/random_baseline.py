"""A random baseline Method that gives random predictions for any input.

Should be applicable to any Setting.
"""
import shlex
from abc import ABC
from dataclasses import dataclass, replace
from typing import ClassVar, Dict, List, Optional, Type, TypeVar, Union

import torch
from simple_parsing import ArgumentParser, mutable_field
from singledispatchmethod import singledispatchmethod
from torch import Tensor
from torch.utils.data import DataLoader

from common.config import Config
from methods.method import Method
from methods.models import HParams, Model, OutputHead
from methods.models.iid_model import IIDModel
from methods.models.class_incremental_model import ClassIncrementalModel
from methods.models.task_incremental_model import TaskIncrementalModel
from methods.task_incremental_method import TaskIncrementalMethod
from settings import IIDSetting, SettingType, TaskIncrementalSetting, ClassIncrementalSetting
from settings.base import EnvironmentBase, Results, Setting
from utils import get_logger

logger = get_logger(__file__)


class RandomOutputHead(OutputHead):
    def forward(self, h_x: Tensor):
        batch_size = h_x.shape[0]
        return torch.rand([batch_size, self.output_size], requires_grad=True).type_as(h_x)

class RandomPredictionsMixin(ABC):
    def encode(self, x: Tensor):
        """ Gives back a random encoding instead of doing a forward pass through
        the encoder.
        """
        batch_size = x.shape[0]
        h_x = torch.rand([batch_size, self.hidden_size])
        return h_x.type_as(x)

    @property
    def output_head_class(self) -> Type[OutputHead]:
        """Property which returns the type of output head to use.

        overwrite this if your model does something different than classification.

        Returns:
            Type[OutputHead]: A subclass of OutputHead.
        """
        return RandomOutputHead

class RandomClassIncrementalModel(RandomPredictionsMixin, ClassIncrementalModel):
    pass

class RandomTaskIncrementalModel(RandomPredictionsMixin, TaskIncrementalModel):
    pass

class RandomIIDModel(RandomPredictionsMixin, IIDModel):
    pass

@dataclass
class RandomBaselineMethod(Method, target_setting=Setting):
    """ Baseline method that gives random predictions for any given setting.

    We do this by creating a base Model with an output head that gives random
    predictions.
    
    TODO: Actually make this compatible with other settings than
    task-incremental and iid. There will probably be some code shuffling to do
    with respect to the `Model` class, as it is moreso aimed at being a `passive`
    Model than an active one at the moment.
    """
    # Configuration options.
    config: Config = mutable_field(Config)

    @singledispatchmethod
    def model_class(self, setting: SettingType) -> Type[Model]:
        raise NotImplementedError(f"No model for setting {setting}!")
    
    @model_class.register
    def _(self, setting: IIDSetting) -> Type[IIDModel]:
        return RandomIIDModel
    
    @model_class.register
    def _(self, setting: ClassIncrementalSetting) -> Type[ClassIncrementalModel]:
        return RandomClassIncrementalModel

    @model_class.register
    def _(self, setting: TaskIncrementalSetting) -> Type[TaskIncrementalModel]:
        return RandomTaskIncrementalModel


if __name__ == "__main__":
    RandomBaselineMethod.main()
