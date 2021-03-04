from typing import List, Optional

import gym
import numpy as np
from sequoia.methods import Method

from .incremental import IncrementalSetting, TestEnvironment


class DummyMethod(Method, target_setting=IncrementalSetting):
    """ Dummy method used to check that the Setting calls `on_task_switch` with the
    right arguments.
    """

    def __init__(self):
        self.n_task_switches = 0
        self.received_task_ids: List[Optional[int]] = []
        self.received_while_training: List[bool] = []

    def fit(self, train_env: gym.Env = None, valid_env: gym.Env = None):
        obs = train_env.reset()
        for i in range(100):
            obs, reward, done, info = train_env.step(train_env.action_space.sample())
            if done:
                break

    def test(self, test_env: TestEnvironment):
        while not test_env.is_closed():
            done = False
            obs = test_env.reset()
            while not done:
                actions = test_env.action_space.sample()
                obs, _, done, info = test_env.step(actions)

    def get_actions(
        self, observations: IncrementalSetting.Observations, action_space: gym.Space
    ):
        return np.ones(action_space.shape)

    def on_task_switch(self, task_id: int = None):
        self.n_task_switches += 1
        self.received_task_ids.append(task_id)
        self.received_while_training.append(self.training)

