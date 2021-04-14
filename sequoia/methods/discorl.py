"""A general baseline Method that just gather learning experience and distill it in a new model.

Should be applicable to any Setting.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import gym
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

from sequoia.methods import register_method
from sequoia.settings import IncrementalRLSetting, TaskIncrementalRLSetting
from sequoia.settings.active.continual import ContinualRLSetting
from sequoia.settings.base import Actions, Environment, Method, Observations
from sequoia.utils import get_logger

from sequoia.methods.stable_baselines3_methods import StableBaselines3Method
from sequoia.methods.stable_baselines3_methods import PPOModel, A2CModel
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

    #Model = PPOModel
    Model = A2CModel

    def configure(self, setting: ContinualRLSetting):
        self.num_inputs = setting.observation_space.x.shape[0]
        self.num_outputs = setting.action_space.n

        # setting.max_steps  # Total training steps (all tasks)
        # setting.steps_per_task  # Total steps per task
        # setting.max_episodes  # Not 100% tested yet
        # setting.episodes_per_task  # Not 100% tested yet

        # empty
        self.buffer = None
        self.criterion = torch.nn.MSELoss()
        self.epoch_distillation = 1


    def fit(self, train_env: gym.Env, valid_env: gym.Env):

        if self.model is None:
            #declare model and student
            self.model = self.create_model(train_env, valid_env)
            self.student = deepcopy(self.model.policy)
            self.opt_student = optim.SGD(params=self.student.parameters(), lr=0.001, momentum=0.9)
        else:
            # TODO: "Adapt"/re-train the model on the new environment.
            self.model.set_env(train_env)

        steps_per_task = self.train_steps_per_task
        self.train_steps_per_task = round(steps_per_task * 0.9)
        # Train the teacher for a portion of the step budget
        super().fit(train_env, valid_env)


        self.train_steps_per_task = steps_per_task

        observation_collection = []
        distill_label_collection = []

        # sample observations and annotate them with teacher
        while len(observation_collection) < self.num_epidodes_saved:
            state = train_env.reset()
            done = False

            while not done:
                # action, _next_state = self.model.predict(state)
                actions, values, log_probs = self.model.policy.forward(state)

                x = torch.as_tensor(state, device=self.model.device)
                logits = torch.as_tensor(log_probs, device=self.model.device)

                observation_collection.append(x)
                distill_label_collection.append(logits)

                state, _, done, _ = train_env.reset.step(actions)

        # concat observations into dataser
        new_observations = TensorDataset(
            torch.cat(observation_collection), torch.cat(distill_label_collection)
        )
        if self.buffer is None:
            self.buffer = new_observations
        else:
            self.buffer = ConcatDataset(self.buffer, new_observations)
        dataloader = DataLoader(self.buffer, batch_size=100, shuffle=False)

        # distillate all knowledge in student
        for epoch in range(self.epoch_distillation):
            for x, y in dataloader:
                self.opt_student.zero_grad()
                actions, values, log_probs = self.student.forward(x)

                loss = self.criterion(actions, y)
                loss.backward()
                self.opt_student.step()



    def get_actions(
        self, observations: ContinualRLSetting.Observations, action_space: gym.Space
    ) -> ContinualRLSetting.Actions:
        # Use the student rather than the Teacher.
        obs = observations.x
        actions, values, log_probs = self.student.forward(obs)
        return actions

    def set_testing(self):
        # todo: quel est le model a set pour l'eval?
        return super().set_testing()


if __name__ == "__main__":
    # setting = IncrementalRLSetting(
    #     dataset="cartpole",
    #     observe_state_directly=True,
    #     nb_tasks=2,
    #     steps_per_task=1_000,
    #     test_steps_per_task=1_000,
    # )
    setting = TaskIncrementalRLSetting(
        dataset="cartpole",
        observe_state_directly=True,
        nb_tasks=2,
        train_task_schedule={
            0: {"gravity": 10, "length": 0.3},
            1000: {"gravity": 10, "length": 0.5},  # second task is 'easier' than the first one.
        },
        max_steps=2000,
    )
    method = DiscoRLMethod()
    results = setting.apply(method)

    # DiscoRLMethod.main()
