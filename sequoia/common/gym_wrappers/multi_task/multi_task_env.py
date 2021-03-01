""" TODO: A special kind of gym.Wrapper that accepts a list of environments or
environment creating functions, and swaps between them at given points in time, or
after each episode.

"""
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import gym
from gym import Wrapper, spaces

from ..utils import IterableWrapper
from .add_task_labels import add_task_labels

EnvOrEnvFn = Union[gym.Env, Callable[..., gym.Env]]


class MultiTaskEnv(IterableWrapper):
    def __init__(self, env: Union[gym.Env, List[EnvOrEnvFn]]):

        self._envs: List[Union[gym.Env], Callable[..., gym.Env]] = []
        # TODO: Should we always require env constructors rather than envs themselves?
        if isinstance(env, gym.Env):
            env = [env]
        for task_env in env:
            self._envs.append(task_env)
        env = self.get_env(0)
        self._current_task_id: int = 0
        self._seeds: List[Optional[int]] = []

        super().__init__(env=env)

        task_label_space = spaces.Discrete(self.nb_tasks)
        self.observation_space = add_task_labels(
            self.env.observation_space, task_label_space
        )

    @property
    def nb_tasks(self) -> int:
        return len(self._envs)

    def get_env(self, env_index: int) -> gym.Env:
        """ Gets the environment at the given index, creating it if necessary.

        If the envs passed to the constructor were constructors rather than gym.Env
        objects, the constructor for the given env will be called.
        """
        if isinstance(self._envs[env_index], gym.Env):
            return self._envs[env_index]
        return self._envs[env_index]()

    def switch_tasks(self, new_task_id: int) -> None:
        assert 0 <= new_task_id < self.nb_tasks

        # TODO: Do we want to close envs on switching tasks? or not?
        # self.env.close()
        self._current_task_id = new_task_id

        self.env = self.get_env(new_task_id)
        # TODO: Assuming the observations/action spaces don't change between tasks.

        if self._seeds and not self._using_live_envs:
            # Seed when creating the env, since we couldn't seed the env instance.
            self.env.seed(self._seeds[self._current_task_id])

        # self.action_space = self.env.action_space
        # self.observation_space = self.env.observation_space
        # self.reward_range = self.env.reward_range
        # self.metadata = self.env.metadata

        # How do we want to manage the 'reset' needed when switching tasks?
        # self._done = True
        # self._info["task_switch"] = True

    def observation(self, observation):
        return add_task_labels(observation, self._current_task_id)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        return obs, reward, done, info

    def reset(self):
        return super().reset()

    def close(self):
        # TODO: Do we close just the current env, or all the envs?
        return super().close()

    def seed(self, seed: Optional[int]):
        # TODO: Do we seed just the current env, or all the envs?
        return self.env.seed()

    def seed_all(
        self, seeds: Union[Optional[int], List[Optional[int]]]
    ) -> List[Optional[int]]:
        """ Seeds all the envs (when possible) and return the seeds used in each. """
        if seeds is None or isinstance(seeds, int):
            seeds = [seeds for _ in range(self.nb_tasks)]

        assert len(seeds) == len(self._envs), "need as many seeds as envs"
        self._seeds = seeds

        if self._using_live_envs:
            results = []
            for env, seed in zip(self._envs, seeds):
                results.append(env.seed(seed))
                env.action_space.seed(seed)
                env.observation_space.seed(seed)
            return results
        else:
            # We can seed the 'live'/current env, but the others may be 'dormant', so we
            # just store the seeds, and we'll use them when 'waking up' the env.
            self._seeds = seeds
            self.seed(seeds[self._current_task_id])
            return self._seeds

    @property
    def _using_live_envs(self) -> bool:
        """Returns True if all the envs passed to this wrapper are 'live' gym.Env
        instances.

        Used internally just to know if we can directly manipulate the envs, or if we
        need to re-create them when using them.

        Returns
        -------
        bool
            True if all envs are 'live' `gym.Env` instances.
        """
        return all(isinstance(env, gym.Env) for env in self._envs)
