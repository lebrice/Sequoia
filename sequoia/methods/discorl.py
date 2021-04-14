"""A general baseline Method that just gather learning experience and distill it in a new model.

Should be applicable to any Setting.
"""
from dataclasses import dataclass
from typing import Optional

import gym
import torch
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

from sequoia.methods import register_method
from sequoia.settings import IncrementalRLSetting, ActiveSetting
from sequoia.settings.base import Actions, Environment, Method, Observations
from sequoia.utils import get_logger

from sequoia.methods.stable_baselines3_methods import StableBaselines3Method
from sequoia.methods.stable_baselines3_methods import PPOMethod, PPOModel
from sequoia.methods.experience_replay import  Buffer
from copy import deepcopy

logger = get_logger(__file__)


@register_method
@dataclass
class DiscoRLMethod(StableBaselines3Method):
    """ Simple method that uses a replay buffer and distillation to reduce forgetting in reinforcement learning.
    """

    Model = PPOModel

    def configure(self, setting: ActiveSetting):
        self.num_inputs = setting.observation_space.x.shape[0]
        self.num_outputs = setting.action_space.n

        # empty
        self.buffer = TensorDataset([torch.tensor(0), torch.tensor(0)])

        # shematic declaration of student
        self.student = deepcopy(self.model.actor_net)

        # we want to sample on policy 10% des episodes et les annoter avec le teacher
        self.num_epidodes_saved = (self.hparams.max_episode_steps * 0.1)
        self.train_steps_per_task = int(self.hparams.max_episode_steps * 0.9) - 1

    def test_prediction(self, observation):
        return self.student(observation)


    def on_task_switch(self, task_id: Optional[int]):
        print(f"Switching from task {self.task} to task {task_id}")


        # from ewc_in_rl class
        observation_collection = []
        distill_label_collection = []
        while len(observation_collection) < self.num_epidodes_saved:
            state = self.model.env.reset()
            for _ in range(1000):
                action = self.get_actions(Observations(state), self.model.env.action_space)
                state, _, done, _ = self.model.env.step(action)
                actions, values, log_probs = self.Model.policy.forward(state)
                observation_collection.append(torch.tensor(state).to(self.model.device))
                distill_label_collection = [torch.tensor(log_probs).to(self.model.device)]
                if done:
                    break

        new_observations = TensorDataset(torch.cat(observation_collection),
                      torch.cat(distill_label_collection))
        self.buffer = ConcatDataset(self.buffer, new_observations)
        dataloader = DataLoader(self.buffer, batch_size=100, shuffle=False)




        # supervised training of student model on the buffer with 10% episodes and teacher annotation
        self.student.fit(dataloader)
        # todo

        if self.training:
            self.task = task_id






if __name__ == "__main__":
    DiscoRLMethod.main()
