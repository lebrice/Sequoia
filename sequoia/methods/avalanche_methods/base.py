from abc import abstractmethod
from typing import List, Optional

import gym
import torch
from avalanche.benchmarks.scenarios import Experience
from avalanche.models import SimpleMLP
from gym import spaces
from gym.spaces.utils import flatdim
from sequoia.common.spaces import Image
from sequoia.methods import Method
from sequoia.settings.passive import (
    PassiveEnvironment,
    PassiveSetting,
    TaskIncrementalSetting,
)
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from .base_strategy import BaseStrategy
from .experience import SequoiaExperience


def environment_to_experience(
    env: PassiveEnvironment, setting: PassiveSetting
) -> Experience:
    """
    TODO: Somehow "convert"  the PassiveEnvironments (dataloaders) from Sequoia 
    into an Experience from Avalanche.
    """
    return SequoiaExperience(env=env, setting=setting)


# TODO: Not sure how to try a setting without task IDs in Avalanche.
class AvalancheMethod(Method, target_setting=TaskIncrementalSetting):
    def __init__(self):
        # Config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def configure(self, setting: TaskIncrementalSetting):
        # model
        self.setting = setting
        self.model: nn.Module = self.create_model(setting)
        # Continual learning strategy
        self.cl_strategy: BaseStrategy = self.create_cl_strategy(setting)

    @abstractmethod
    def create_model(self, setting: TaskIncrementalSetting) -> nn.Module:
        image_space: Image = setting.observation_space.x
        input_dims = flatdim(image_space)
        assert isinstance(
            setting.action_space, spaces.Discrete
        ), "assume a classification problem for now."
        num_classes = setting.action_space.n
        return SimpleMLP(input_size=input_dims, num_classes=num_classes).to(self.device)

    @abstractmethod
    def create_cl_strategy(self, setting: TaskIncrementalSetting) -> BaseStrategy:
        self.optimizer = SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.criterion = CrossEntropyLoss()
        from .naive import Naive
        return Naive(
            self.model,
            self.optimizer,
            self.criterion,
            train_mb_size=1,
            train_epochs=2,
            eval_mb_size=1,
            device=self.device,
            eval_every=0,
        )

    def fit(self, train_env: PassiveEnvironment, valid_env: PassiveEnvironment):
        train_exp = environment_to_experience(train_env, setting=self.setting)
        valid_exp = environment_to_experience(valid_env, setting=self.setting)
        self.cl_strategy.train(train_exp, eval_streams=[valid_exp], num_workers=0)
        # return super().fit(train_env, valid_env)

    def test(self, test_env: PassiveEnvironment):
        self.cl_strategy.test_epoch(test_env)

    def get_actions(
        self, observations: TaskIncrementalSetting.Observations, action_space: gym.Space
    ) -> TaskIncrementalSetting.Actions:
        # TODO: There doesn't seem to be a way of doing this 'eval epoch'

        # TODO: Perform inference with the model.
        with torch.no_grad():
            logits = self.model(observations.x.to(self.device))
            y_pred = logits.argmax(-1)
            return self.target_setting.Actions(y_pred=y_pred)

    def on_task_switch(self, task_id: Optional[int]) -> None:
        # TODO: How do we let the cl_strategy know?
        if self.training:
            # No need to tell the cl_strategy, because we call `.train` which calls
            # `before_training_exp` with the current exp (the current task)
            pass
        else:
            # During test-time, there might be a task boundary, and we need to let the
            # cl_strategy and the plugins know.
            # TODO: Get this working, figure out what the plugins expect to retrieve
            # from the cl_strategy in this callback.
            if self.cl_strategy.experience is not None:
                self.cl_strategy.after_eval_exp()
                self.cl_strategy.before_eval_exp()
            
        self._was_training = self.training