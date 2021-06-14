import gym
from abc import ABC, abstractmethod
from typing import List, Sequence, Optional
from sequoia.common.gym_wrappers import IterableWrapper
import numpy as np
from gym import spaces
from typing import Callable
from sequoia.common.gym_wrappers.multi_task_environment import add_task_labels
from sequoia.utils.logging_utils import get_logger
logger = get_logger(__file__)


class MultiEnvWrapper(IterableWrapper, ABC):
    """ TODO: Wrapper like that iterates over the envs.

    Could look a little bit like this:
    https://github.com/rlworkgroup/garage/blob/master/src/garage/envs/multi_env_wrapper.py
    """
    def __init__(self, envs: List[gym.Env], add_task_ids: bool = False):
        self._envs = envs
        self._current_task_id = 0
        self.nb_tasks = len(envs)
        self._envs_is_closed: Sequence[bool] = np.zeros([self.nb_tasks], dtype=bool)
        self._add_task_labels = add_task_ids
        self.rng: np.random.Generator = np.random.default_rng()
        super().__init__(env=self._envs[self._current_task_id])
        self.task_label_space = spaces.Discrete(self.nb_tasks)
        if self._add_task_labels:
            self.observation_space = add_task_labels(self.env.observation_space, self.task_label_space)

    def set_task(self, task_id: int) -> None:
        self._current_task_id = task_id
        super().__init__(env=self._envs[self._current_task_id])
        if self._add_task_labels:
            self.observation_space = add_task_labels(self.env.observation_space, self.task_label_space)

    @abstractmethod
    def next_task(self) -> int:
        pass

    def reset(self):
        if all(self._envs_is_closed):
            self.close(close_all=True)
        elif self.env.is_closed():
            self._envs_is_closed[self._current_task_id] = True
        self.set_task(self.next_task())
        obs = super().reset()
        if self._add_task_labels:
            obs = add_task_labels(obs, self._current_task_id)
        return obs

    def step(self, action):
        obs, rewards, done, info = super().step(action)
        if self._add_task_labels:
            obs = add_task_labels(obs, self._current_task_id)
        return obs, rewards, done, info

    def close(self, close_all: bool = True) -> None:
        """ Close the environment for the current task, or of all tasks if `close_all`
        is passed.
        """
        if close_all:
            logger.info(f"Closing all envs")
            for env_index, (env_envs_is_closed, env) in enumerate(zip(self._envs_is_closed, self._envs)):
                if not env_envs_is_closed:
                    self._envs_is_closed[env_index] = True
                    env.close()
        else:
            if self._envs_is_closed[self._current_task_id]:
                raise RuntimeError(f"Can't close the same env twice..")
            self._envs_is_closed[self._current_task_id] = True

        if not all(self._envs_is_closed):
            self.env.close()
        else:
            # Close 'for real'?
            super().close()

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Sets the seed for this env's random number generator(s).

        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.

        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
            number generators. The first value in the list should be the
            "main" seed, or the value which a reproducer should pass to
            'seed'. Often, the main seed equals the provided 'seed', but
            this won't be true if seed=None, for example.
        """
        self.rng = np.random.default_rng(seed) 
        seeds = [seed]
        for index, env in enumerate(self._envs):
            env_seeds: Optional[List[int]] = env.seed(seed + index)
            seeds.extend(env_seeds or [])
        return seeds


class ConcatEnvsWrapper(MultiEnvWrapper):
    """ Wrapper that exhausts the current environment before moving onto the next. """ 
    def next_task(self) -> int:
        assert not all(self._envs_is_closed)
        if not self._envs_is_closed[self._current_task_id]:
            return self._current_task_id
        # TODO: Close the env when we reach the end? or leave that up to the wrapper?
        return (self._current_task_id + 1) % self.nb_tasks


class RoundRobinWrapper(MultiEnvWrapper):
    def __init__(self, envs, add_task_ids=False):
        super().__init__(envs, add_task_ids=add_task_ids)
        self._current_task_id = -1

    def next_task(self) -> int:
        assert not all(self._envs_is_closed)
        next_task = (self._current_task_id + 1) % self.nb_tasks
        while self._envs_is_closed[next_task]:
            next_task += 1
            next_task %= self.nb_tasks
        return next_task


class RandomMultiEnvWrapper(MultiEnvWrapper):
    def next_task(self) -> int:
        assert not all(self._envs_is_closed)
        available_ids = np.arange(self.nb_tasks)[~self._envs_is_closed].tolist()
        return self.rng.choice(available_ids)
