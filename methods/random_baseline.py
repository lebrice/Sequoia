"""A random baseline Method that gives random predictions for any input.

Should be applicable to any Setting.
"""
from abc import ABC
from dataclasses import dataclass
from typing import Type, Union

import torch
from torch import Tensor

from methods.method import Method
from methods.models import Agent, Model, OutputHead
from methods.models.iid_model import IIDModel
from methods.models.task_incremental_model import TaskIncrementalModel
from methods.models.model_addons import ClassIncrementalModel as ClassIncrementalModelMixin
from settings import (ActiveSetting, ClassIncrementalSetting, IIDSetting,
                      RLSetting, Setting, SettingType, TaskIncrementalSetting)
from utils import get_logger, singledispatchmethod

from .models import Model
from .models.agent import Agent
from .models.random_agent import RandomAgent

logger = get_logger(__file__)


class RandomOutputHead(OutputHead):
    def forward(self, h_x: Tensor):
        batch_size = h_x.shape[0]
        return torch.rand([batch_size, self.output_size], requires_grad=True).type_as(h_x)

class RandomPredictionsMixin(ABC):
    """ A mixin class that when applied to a Model class makes it give random
    predictions.
    """
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


class RandomClassIncrementalModel(RandomPredictionsMixin, Model):
    pass


class RandomTaskIncrementalModel(RandomPredictionsMixin, TaskIncrementalModel):
    pass


class RandomIIDModel(RandomPredictionsMixin, IIDModel):
    pass



# class RandomAgent(RandomPredictionsMixin, Agent):
#     pass


def get_random_model_class(base_model_class: Type[Model]) -> Type[Union[RandomPredictionsMixin, Model]]:
    class RandomModel(RandomPredictionsMixin, base_model_class):
        pass
    return RandomModel

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

    @singledispatchmethod
    def model_class(self, setting: SettingType) -> Type[Model]:
        raise NotImplementedError(f"No known model for setting of type {type(setting)} (registry: {self.model_class.registry})")
    
    @model_class.register
    def _(self, setting: ActiveSetting) -> Type[Agent]:
        # TODO: Make a 'random' RL method.
        return RandomAgent

    @model_class.register
    def _(self, setting: ClassIncrementalSetting) -> Type[ClassIncrementalModelMixin]:
        # IDEA: Generate the model dynamically instead of creating one of each.
        # (This doesn't work atm because super() gives back a Model)
        # return get_random_model_class(super().model_class(setting))
        return RandomClassIncrementalModel

    @model_class.register
    def _(self, setting: TaskIncrementalSetting) -> Type[TaskIncrementalModel]:
        return RandomTaskIncrementalModel

    @model_class.register
    def _(self, setting: IIDSetting) -> Type[IIDModel]:
        return RandomIIDModel


if __name__ == "__main__":
    RandomBaselineMethod.main()
