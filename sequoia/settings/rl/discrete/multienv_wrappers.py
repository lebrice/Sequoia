""" Wrappers that around multiple environments.

These wrappers can be used to get different kinds of multi-task environments, or even to
concatenate environments.
"""
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Sequence, Union

import gym
import numpy as np
from gym import spaces
from sequoia.common.gym_wrappers import IterableWrapper
from sequoia.common.gym_wrappers.multi_task_environment import add_task_labels
from sequoia.common.gym_wrappers.utils import MayCloseEarly
from sequoia.utils.generic_functions import concatenate
from sequoia.utils.logging_utils import get_logger

logger = get_logger(__file__)


def instantiate_env(env: Union[str, gym.Env, Callable[[], gym.Env]]) -> gym.Env:
    if isinstance(env, gym.Env):
        return env
    if isinstance(env, str):
        return gym.make(env)
    assert callable(env)
    return env()


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

        self._instantiate_env(self._current_task_id)
        super().__init__(env=self._envs[self._current_task_id])
        self.task_label_space = spaces.Discrete(self.nb_tasks)
        if self._add_task_labels:
            self.observation_space = add_task_labels(
                self.env.observation_space, self.task_label_space
            )

    def _instantiate_env(self, index: int) -> None:
        self._envs[index] = instantiate_env(self._envs[index])

    def set_task(self, task_id: int) -> None:
        if self.is_closed(env_index=None):
            raise gym.error.ClosedEnvironmentError(
                f"Can't call set_task on the env, since it's already closed."
            )
        self._current_task_id = task_id
        # Use super().__init__() to reset the `self.env` attribute in gym.Wrapper.
        # TODO: This also resets the '_is_closed' on self.
        # TODO: This resets the 'observation_' and 'action_' etc objects that are saved
        # in the constructor of the 'IterableWrapper'
        self._instantiate_env(self._current_task_id)
        gym.Wrapper.__init__(self, env=self._envs[self._current_task_id])
        if self._add_task_labels:
            self.observation_space = add_task_labels(
                self.env.observation_space, self.task_label_space
            )

    @abstractmethod
    def next_task(self) -> int:
        pass

    def reset(self):
        if all(self._envs_is_closed):
            self.close()
        elif isinstance(self.env, MayCloseEarly) and self.env.is_closed():
            self._envs_is_closed[self._current_task_id] = True
        self.set_task(self.next_task())
        obs = super().reset()
        return self.observation(obs)

    def step(self, action):
        obs, rewards, done, info = super().step(action)
        obs = self.observation(obs)
        return obs, rewards, done, info

    def is_closed(self, env_index: int = None):
        """ returns `True` if the environment at index `env_index` is closed, otherwise
        if `env_index` is None, returns `True` if `close()` was called on the wrapper.
        (todo: or if all envs are closed.)
        """
        if env_index is None:
            # Return wether this wrapper itself was closed manually (from outside).
            # TODO: Should we also check if all envs are closed? If so, should we close
            # this env manually?
            if self._is_closed:
                return True
            elif all(self.is_closed(env_id) for env_id in range(self.nb_tasks)):
                self.close(env_index=None)
                return True
            return False

        assert isinstance(env_index, int)
        # Return wether the env at that index is closed.
        if isinstance(self._envs[env_index], MayCloseEarly):
            env_is_closed = self._envs[env_index].is_closed()
            # NOTE: These shouls always be the same, but just in case:
            self._envs_is_closed[env_index] = env_is_closed
        return self._envs_is_closed[env_index]

    def close(self, env_index: int = None) -> None:
        """ Close the environment for the given index, or of all envs if `env_index` is
        `None`.
        """
        if env_index is None:
            logger.info(f"Closing all envs")
            for env_index, (env_is_closed, env) in enumerate(
                zip(self._envs_is_closed, self._envs)
            ):
                if not env_is_closed:
                    self._envs_is_closed[env_index] = True
                    env.close()
            # BUG: Not sure why this is actually causing a recursion error.. The idea
            # was to call `MayCloseEarly.close()`.
            # super().close()
            self._is_closed = True
        else:
            if self._envs_is_closed[env_index]:
                raise RuntimeError(f"Env at index {env_index} is already closed...")
            self._envs_is_closed[env_index] = True
            self._envs[env_index].close()

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
        env_seeds = self.rng.integers(0, 1e8, size=len(self._envs)).tolist()
        seeds = env_seeds.copy()
        for index, env_seed in enumerate(env_seeds):
            self._instantiate_env(index)
            env = self._envs[index]
            env_seeds: Optional[List[int]] = env.seed(env_seed)
            seeds.extend(env_seeds or [])
        return seeds

    def observation(self, observation):
        if self._add_task_labels:
            return add_task_labels(observation, task_labels=self._current_task_id)
        return observation


from sequoia.common.gym_wrappers.env_dataset import EnvDataset
from torch.utils.data import ChainDataset


class ConcatEnvsWrapper(MultiEnvWrapper):
    """ Wrapper that exhausts the current environment before moving onto the next. """

    def __init__(
        self,
        envs: List[gym.Env],
        add_task_ids: bool = False,
        on_task_switch_callback: Callable[[Optional[int]], Any] = None,
    ):
        super().__init__(envs, add_task_ids=add_task_ids)
        self.on_task_switch_callback = on_task_switch_callback

    def reset(self):
        old_task = self._current_task_id
        observation = super().reset()
        new_task = self._current_task_id
        if old_task != new_task and self.on_task_switch_callback:
            self.on_task_switch_callback(new_task if self._add_task_labels else None)
        return observation

    def next_task(self) -> int:
        assert not all(self._envs_is_closed)
        if not self._envs_is_closed[self._current_task_id]:
            return self._current_task_id
        # TODO: Close the env when we reach the end? or leave that up to the wrapper?
        return (self._current_task_id + 1) % self.nb_tasks

    def __iter__(self):
        # BUG: iterating over a MultiEnvWrapper
        return super().__iter__()

    def send(self, action):
        return super().send(action)

    #     if not self.is_closed(env_id):
    #         self.observation_ = env.reset()
    #         return super().__iter__(self)
    #     for env_id, env in enumerate(self._envs):

    # yield from super().__iter__()
    # self.observation_ = self.reset()


# Register this as a 'concat' handler for gym environments!


@concatenate.register(gym.Env)
def _concatenate_gym_envs(
    first_env: gym.Env, *other_envs: gym.Env
) -> ConcatEnvsWrapper:
    return ConcatEnvsWrapper([first_env, *other_envs])


class RoundRobinWrapper(MultiEnvWrapper):
    """ MultiEnvWrapper that alternates between the non-closed environments in a
    round-robin fashion.
    """

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


class CustomMultiEnvWrapper(MultiEnvWrapper):
    """ MultiEnvWrapper that uses a custom callable to determine which env to use next.
    """

    def __init__(
        self,
        envs: List[gym.Env],
        add_task_ids: bool = False,
        custom_new_task_fn: Callable[[MultiEnvWrapper], int] = None,
    ):
        super().__init__(envs, add_task_ids=add_task_ids)
        assert custom_new_task_fn, "Must pass a custom function to this wrapper."
        self._custom_new_task_fn = custom_new_task_fn

    def next_task(self):
        return self._custom_new_task_fn
        return super().next_task()
