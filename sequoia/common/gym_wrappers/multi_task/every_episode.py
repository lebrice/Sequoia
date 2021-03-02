""" Wrapper which switches between tasks after each episode. """
import random
from typing import Dict, List, Optional, Union

import gym
import numpy as np
from gym import spaces

from .multi_task_env import EnvOrEnvFn, MultiTaskEnv


class ChangeTaskAfterEachEpisode(MultiTaskEnv):
    def __init__(self, env: Union[gym.Env, List[EnvOrEnvFn]]):
        super().__init__(env)
        self.rng: np.random.Generator = np.random.default_rng(seed=None)

    # def step(self, action):
    #     return super().step(action)

    def new_task(self) -> int:
        # Returns the new task to switch to.
        next_task_index = self.rng.choice(range(self.nb_tasks))
        next_task_index = int(next_task_index)  # np.int64 -> int.
        assert isinstance(next_task_index, int)
        return next_task_index

    def reset(self):
        next_task_index = self.new_task()
        self.switch_tasks(new_task_index=next_task_index)
        return super().reset()

    def seed(self, seed: Optional[int]) -> List[int]:
        # TODO: Not sure how to get the seed from the RNG object (when seed is `None`).
        self.rng = np.random.default_rng(seed=seed)
        return super().seed(seed)


class RoundRobinEnv(ChangeTaskAfterEachEpisode):
    """ Multi-Task Environment, that switches environments after each episode,
    incrementally, until all environments are exhausted or closed.
    """

    def __init__(self, envs: Union[gym.Env, List[EnvOrEnvFn]]):
        super().__init__(envs)

    def new_task(self) -> int:
        new_task_index = (self._current_task_index + 1) % self.nb_tasks
        # TODO: Check if the envs are closed or have reached some kind of limit?
        if self._using_live_envs:
            pass
        return new_task_index

    def reset(self):
        new_task_index = (self._current_task_index + 1) % self.nb_tasks
        self.switch_tasks(new_task_index=new_task_index)
        return super().reset()
