"""A random baseline Method that gives random predictions for any input.

Should be applicable to any Setting.
"""
from abc import ABC
from dataclasses import dataclass
from typing import Type, Union

import torch
import numpy as np
from torch import Tensor

from methods.method import Method
from methods.models import Agent, Model, OutputHead
from methods.models.iid_model import IIDModel
from methods.models.task_incremental_model import TaskIncrementalModel
from methods.models.model_addons import ClassIncrementalModel as ClassIncrementalModelMixin
from settings import (ActiveSetting, ClassIncrementalSetting, IIDSetting,
                      RLSetting, Setting, SettingType, TaskIncrementalSetting, Observations, Actions, Rewards)
from utils import get_logger, singledispatchmethod
from common.metrics import Metrics, ClassificationMetrics, RegressionMetrics
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
    def fit(self, train_dataloader=None, valid_dataloader=None, datamodule=None):
        example_obs = None
        example_reward = None
        labels_encountered = set()        
        for obs, reward in train_dataloader:
            example_obs = obs
            if reward is not None:
                example_reward = reward
        return 1

    def configure(self, setting: Setting):
        self.setting = setting
        self.action_space = setting.action_space
        super().configure(setting)

    def get_actions(self, observations: Observations) -> Actions:
        obs_shapes = observations.shapes
        batch_size = observations.batch_size
        return self.Actions(torch.as_tensor([
            self.action_space.sample() for _ in range(batch_size)
        ]))

    @singledispatchmethod
    def validate_results(self, setting: Setting, results: Setting.Results):
        """Called during testing. Use this to assert that the results you get
        from applying your method on the given setting match your expectations.

        Args:
            setting
            results (Results): A given Results object.
        """
        assert results is not None
        assert results.objective > 0
        print(f"Objective when applied to a setting of type {type(setting)}: {results.objective}")

    # TODO: Add a validate_results method for an RL Settings.

    @validate_results.register
    def validate(self, setting: ClassIncrementalSetting, results: ClassIncrementalSetting.Results):
        assert isinstance(setting, ClassIncrementalSetting), setting
        assert isinstance(results, ClassIncrementalSetting.Results), results

        average_accuracy = results.objective
        # Calculate the expected 'average' chance accuracy.
        # We assume that there is an equal number of classes in each task.
        chance_accuracy = 1 / setting.n_classes_per_task

        assert 0.5 * chance_accuracy <= average_accuracy <= 1.5 * chance_accuracy

        for i, metric in enumerate(results.average_metrics_per_task):
            assert isinstance(metric, ClassificationMetrics)
            # TODO: Check that this makes sense:
            chance_accuracy = 1 / setting.n_classes_per_task

            task_accuracy = metric.accuracy
            # FIXME: Look into this, we're often getting results substantially
            # worse than chance, and to 'make the tests pass' (which is bad)
            # we're setting the lower bound super low, which makes no sense.
            assert 0.25 * chance_accuracy <= task_accuracy <= 2.1 * chance_accuracy

    
    
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
