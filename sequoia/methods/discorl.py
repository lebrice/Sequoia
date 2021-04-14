"""A general baseline Method that just gather learning experience and distill it in a new model.

Should be applicable to any Setting.
"""
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Type, Any
from argparse import ArgumentParser, Namespace

import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import tqdm
from torch import Tensor
from torchvision.models import ResNet
from wandb.wandb_run import Run

from sequoia.methods import register_method
from sequoia.settings import IncrementalRLSetting, ActiveSetting
from sequoia.settings.base import Actions, Environment, Method, Observations
from sequoia.utils import get_logger

from sequoia.methods.stable_baselines3_methods import PPOMethod, PPOModel
from sequoia.methods.experience_replay import  Buffer

logger = get_logger(__file__)


@register_method
@dataclass
class DiscoRLMethod(Method, target_setting=IncrementalRLSetting):
    """ Simple method that uses a replay buffer and distillation to reduce forgetting in reinforcement learning.
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        buffer_capacity: int = 200,
        max_epochs_per_task: int = 10,
        weight_decay: float = 1e-6,
        seed: int = None,
    ):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.buffer_capacity = buffer_capacity

        self.net: ResNet
        self.buffer: Optional[Buffer] = None
        self.optim: torch.optim.Optimizer
        self.task: int = 0
        self.rng = np.random.RandomState(seed)
        self.seed = seed
        if seed:
            torch.manual_seed(seed)
            torch.set_deterministic(True)

        self.epochs_per_task: int = max_epochs_per_task
        self.early_stop_patience: int = 2

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def configure(self, setting: ActiveSetting):
        self.num_inputs = setting.observation_space.x.shape[0]
        self.num_outputs = setting.action_space.n

        # buffer that should be extendable
        self.buffer = self.buffer = Buffer(
                capacity=self.buffer_capacity,
                input_shape=setting.shape,
                extra_buffers={"t": torch.LongTensor},
                rng=self.rng,
            ).to(device=self.device)

        # model that will solve the policy without further intervention
        self.model = PPOModel(
            policy="CnnPolicy",
            env=setting
        )

    def fit(self, train_env: Environment, valid_env: Environment):
        assert isinstance(train_env, gym.Env)  # Just to illustrate that it's a gym Env.


        # train PPO sur 90% de episodes

        # sample on policy 10% des episodes et les annoter avec PPO



    def on_task_switch(self, task_id: Optional[int]):
        print(f"Switching from task {self.task} to task {task_id}")

        # supervised training of a model with same architecture than the actor on the saved samples

        # todo

        if self.training:
            self.task = task_id






if __name__ == "__main__":
    DiscoRLMethod.main()
