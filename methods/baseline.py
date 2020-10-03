import itertools
import shlex
from abc import ABC
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import ClassVar, Dict, List, Optional, Type, TypeVar, Union

import torch
from common.config import Config
from common.loss import Loss
from pytorch_lightning import LightningModule, Trainer
from settings import (ClassIncrementalSetting, ContinualRLSetting,
                      EnvironmentBase, IIDSetting, Results, RLSetting, Setting,
                      SettingType, TaskIncrementalSetting)
from simple_parsing import ArgumentParser, mutable_field, subparsers

from torch import Tensor
from torch.utils.data import DataLoader
from utils import get_logger, singledispatchmethod

from .class_incremental_method import ClassIncrementalMethod
from .method import Method
from .models import Model, OutputHead
from .models.actor_critic_agent import ActorCritic
from .models.agent import Agent
from .models.iid_model import IIDModel
from .models.task_incremental_model import TaskIncrementalModel
from .models.model_addons import ClassIncrementalModel as ClassIncrementalModelMixin
logger = get_logger(__file__)


@dataclass
class BaselineMethod(Method, target_setting=Setting):
    """ TODO: Does it even make sense to have this? It basically just delegates
    to the different 'default' method classes for each setting atm
    """
    @singledispatchmethod
    def model_class(self, setting: SettingType) -> Type[Model]:
        raise NotImplementedError(f"No model registered for setting {setting}!")
    
    @model_class.register
    def _(self, setting: ClassIncrementalSetting) -> Type[ClassIncrementalModelMixin]:
        return Model

    @model_class.register
    def _(self, setting: TaskIncrementalSetting) -> Type[TaskIncrementalModel]:
        return TaskIncrementalModel

    @model_class.register
    def _(self, setting: IIDSetting) -> Type[IIDModel]:
        return IIDModel

    @model_class.register
    def _(self, setting: ContinualRLSetting) -> Type[Agent]:
        return ActorCritic

    @model_class.register
    def _(self, setting: RLSetting) -> Type[Agent]:
        return ActorCritic
    
    def on_task_switch(self, task_id: int) -> None:
        self.model.on_task_switch(task_id)

    @singledispatchmethod
    def train(self, setting: Setting):
        return super().train(setting)

if __name__ == "__main__":
    BaselineMethod.main()
