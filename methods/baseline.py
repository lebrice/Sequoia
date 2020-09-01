import itertools
import shlex
from abc import ABC
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import ClassVar, Dict, List, Optional, Type, TypeVar, Union

import torch
from pytorch_lightning import LightningModule, Trainer
from singledispatchmethod import singledispatchmethod
from torch import Tensor
from torch.utils.data import DataLoader

from common.config import Config
from common.loss import Loss
from settings import (ClassIncrementalSetting, IIDSetting, SettingType,
                      TaskIncrementalSetting)
from settings.base import EnvironmentBase, Results, Setting
from simple_parsing import ArgumentParser, mutable_field, subparsers
from utils import get_logger

from .method import Method
from .models import HParams, Model, OutputHead
from .models.iid_model import IIDModel
from .models.task_incremental_model import TaskIncrementalModel
from .models.class_incremental_model import ClassIncrementalModel


logger = get_logger(__file__)

@dataclass
class BaselineMethod(Method, target_setting=Setting):
    """ Baseline method that gives random predictions for any given setting.

    We do this by creating a base Model with an output head that gives random
    predictions.
    
    TODO: Actually make this compatible with other settings than
    task-incremental and iid. There will probably be some code shuffling to do
    with respect to the `Model` class, as it is moreso aimed at being a `passive`
    Model than an active one.
    """
    @singledispatchmethod
    def model_class(self, setting: SettingType) -> Type[Model]:
        raise NotImplementedError(f"No model registered for setting {setting}!")
    
    @model_class.register
    def _(self, setting: IIDSetting) -> Type[IIDModel]:
        return IIDModel
    
    @model_class.register
    def _(self, setting: ClassIncrementalSetting) -> Type[ClassIncrementalModel]:
        return ClassIncrementalModel

    @model_class.register
    def _(self, setting: TaskIncrementalSetting) -> Type[TaskIncrementalModel]:
        return TaskIncrementalModel
    
    def on_task_switch(self, task_id: int) -> None:
        self.model.on_task_switch(task_id)
    
if __name__ == "__main__":
    BaselineMethod.main()
