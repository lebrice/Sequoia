"""A general baseline Method that just gather learning experience and distill it in a new model.

Should be applicable to any Setting.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import gym
import torch
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

from sequoia.methods import register_method
from sequoia.settings import IncrementalRLSetting, ActiveSetting
from sequoia.settings.active.continual import ContinualRLSetting
from sequoia.settings.base import Actions, Environment, Method, Observations
from sequoia.utils import get_logger

from sequoia.methods.stable_baselines3_methods import StableBaselines3Method
from sequoia.methods.stable_baselines3_methods import PPOMethod, PPOModel
from sequoia.methods.experience_replay import Buffer
from copy import deepcopy
from sequoia.utils.categorical import Categorical


logger = get_logger(__file__)


@register_method
@dataclass
class DiscoRLMethod(StableBaselines3Method):
    """ Simple method that uses a replay buffer and distillation to reduce forgetting in
    reinforcement learning.

    [WIP] IDEA:
    - teacher model: trained online on the environment.
    - student model: trained with some kind of immitation learning / behaviour cloning
    """

    Model = PPOModel

    def configure(self, setting: ContinualRLSetting):
        self.num_inputs = setting.observation_space.x.shape[0]
        self.num_outputs = setting.action_space.n

        setting.max_steps  # Total training steps (all tasks)
        setting.steps_per_task  # Total steps per task
        setting.max_episodes  # Not 100% tested yet
        setting.episodes_per_task  # Not 100% tested yet

        # empty
        self.buffer = TensorDataset([torch.tensor(0), torch.tensor(0)])

        # shematic declaration of student
        self.student = deepcopy(self.model.policy)

        # IDEA: Pretend that the setting has fewer steps than it actually doe, just for
        # the call to `configure`.
        # super().configure(setting)

        # we want to sample on policy 10% des episodes et les annoter avec le teacher
        self.num_epidodes_saved = self.hparams.max_episode_steps * 0.1
        self.train_steps_per_task = int(self.hparams.max_episode_steps * 0.9) - 1

    def fit(self, train_env: gym.Env, valid_env: gym.Env):
        steps_per_task = self.train_steps_per_task
        self.train_steps_per_task = round(steps_per_task * 0.9)
        # Train the teacher for a portion of the step budget
        super().fit(train_env, valid_env)

        self.train_steps_per_task = steps_per_task

        observation_collection = []
        distill_label_collection = []
        previous_task_env: gym.Env = self.model.env

        while len(observation_collection) < self.num_epidodes_saved:
            state = previous_task_env.reset()
            done = False

            while not done:
                # action, _next_state = self.model.predict(state)
                actions, values, log_probs = self.model.policy.forward(state)

                x = torch.as_tensor(state, device=self.model.device)
                logits = torch.as_tensor(log_probs, device=self.model.device)

                observation_collection.append(x)
                distill_label_collection.append(logits)

                state, _, done, _ = previous_task_env.step(actions)

        new_observations = TensorDataset(
            torch.cat(observation_collection), torch.cat(distill_label_collection)
        )
        self.buffer = ConcatDataset(self.buffer, new_observations)
        dataloader = DataLoader(self.buffer, batch_size=100, shuffle=False)

        for x, y in dataloader:
            actions, values, log_probs = self.student.forward(state)
            # TODO

    def get_actions(
        self, observations: ContinualRLSetting.Observations, action_space: gym.Space
    ) -> ContinualRLSetting.Actions:
        # TODO: Use the student rather than the Teacher.
        return super().get_actions(observations, action_space)

    def set_testing(self):
        return super().set_testing()

    def test_prediction(self, observation):
        return self.student(observation)

    def on_task_switch(self, task_id: Optional[int]):
        print(f"Switching from task {self.task} to task {task_id}")
        if task_id == 0:
            return
        # supervised training of student model on the buffer with 10% episodes and teacher annotation
        self.student.fit(dataloader)
        # todo

        if self.training:
            self.task = task_id


if __name__ == "__main__":
    setting = IncrementalRLSetting(
        dataset="cartpole",
        observe_state_directly=True,
        nb_tasks=2,
        steps_per_task=1_000,
        test_steps_per_task=1_000,
    )
    method = DiscoRLMethod()
    results = setting.apply(method)

    # DiscoRLMethod.main()
