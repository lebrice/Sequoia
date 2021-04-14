"""A general baseline Method that just gather learning experience and distill it in a new model.

Should be applicable to any Setting.
"""
from dataclasses import dataclass
from typing import Optional

import gym
import torch

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
        dix_pourcent_episode=None # todo
        self.buffer.add(dix_pourcent_episode)




    def on_task_switch(self, task_id: Optional[int]):
        print(f"Switching from task {self.task} to task {task_id}")

        # supervised training of a model with same architecture than the actor on the saved samples

        # todo

        if self.training:
            self.task = task_id






if __name__ == "__main__":
    DiscoRLMethod.main()
