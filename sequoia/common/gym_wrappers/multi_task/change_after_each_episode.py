""" Wrapper which switches between tasks after each episode. """
import random
from typing import List, Optional, Union

import gym
import numpy as np
import random
from .multi_task_env import EnvOrEnvFn, MultiTaskEnv


class ChangeAfterEachEpisode(MultiTaskEnv):
    def __init__(self, env: Union[gym.Env, List[EnvOrEnvFn]]):
        super().__init__(env)
        self.rng: np.random.Generator = np.random.default_rng(seed=None)

    def step(self, action):
        obs, rewards, done, info = super().step(action)
        return obs, rewards, done, info

    def reset(self):
        next_task_id = self.rng.choice(range(self.nb_tasks))
        next_task_id = int(next_task_id) # np.int64 -> int.
        assert isinstance(next_task_id, int)
        self.switch_tasks(new_task_id=next_task_id)
        return super().reset()

    def seed(self, seed: Optional[int]) -> List[int]:
        # TODO: Not sure how to get the seed from the RNG object (when seed is `None`).
        self.rng = np.random.default_rng(seed=seed)
        return super().seed(seed)

    def __iter__(self):
        obs = self.reset()
        yield obs
        for batch in super().__iter__():
            yield self.batch(batch)
