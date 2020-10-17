"""TODO: Mix of AsyncVectorEnv and SyncVectorEnv, where we have a series of
environments on each worker.
"""
import multiprocessing as mp
import itertools
from functools import partial
from typing import Any, Callable, Iterable, List, Sequence, Tuple, TypeVar, Union

import gym
import numpy as np
from gym import spaces
from gym.vector.sync_vector_env import SyncVectorEnv

from .async_vector_env import AsyncVectorEnv
from utils.utils import n_consecutive

from ..utils import space_with_new_shape

T = TypeVar("T")


class AsyncChunkedVectorEnv(AsyncVectorEnv):
    def __init__(self,
                 env_fns,
                 context=None,
                 worker=None,
                 shared_memory=True,
                 n_workers: int = None,
                 **kwargs):
        self.batch_size = len(env_fns)

        if n_workers is None:
            n_workers = mp.cpu_count()
        self.n_workers: int = n_workers
        if self.batch_size < self.n_workers:
            self.n_workers = self.batch_size
        self.max_envs_per_worker = int(self.batch_size / self.n_workers)
        self.min_envs_per_worker = self.batch_size // self.n_workers

        groups: List[List[Callable[[], gym.Env]]] = [
            [] for _ in range(self.n_workers)
        ]
        for i, env_fn in enumerate(env_fns):
            groups[i % self.n_workers].append(env_fn)
        
        # groups = list(n_consecutive(env_fns, self.max_envs_per_worker))
        assert len(groups) == self.n_workers
        # Lengths of each group.
        self.group_lengths = list(map(len, groups))

        new_env_fns: List[Callable[[], gym.Env]] = [
            partial(SyncVectorEnv, env_fns_group) for env_fns_group in groups
        ]
        super().__init__(
            new_env_fns,
            context=context,
            worker=worker,
            shared_memory=shared_memory,
            **kwargs
        )
        # TODO: Unbatch / flatten the observations/actions/reward spaces?
        # here? or in a wrapper?
        self.observation_space: gym.Space
        self.action_space: gym.Space
        # Keep a copy of the spaces with the 'extra dims' included, so we can
        # use them to reshape stuff later.
        self.observation_space_extra_dim = self.observation_space
        self.action_space_extra_dim = self.action_space

        self.observation_space = remove_extra_batch_dim_from_space(self.observation_space)
        self.action_space = remove_extra_batch_dim_from_space(self.action_space)
        self.reset()

        assert not hasattr(self, "reward_space")
        self.reward_space = spaces.Tuple([
            spaces.Box(low=self.reward_range[0], high=self.reward_range[1], shape=())
            for _ in range(self.batch_size)
        ])

    def reset(self):
        return self.observation(super().reset())

    def step(self, action):
        observation, reward, done, info = super().step(self.action(action))
        done = np.array(list(itertools.chain.from_iterable(done)))
        info = list(itertools.chain.from_iterable(info))
        return self.observation(observation), self.reward(reward), done, info

    def action(self, action: Tuple[T]) -> Tuple[Tuple[T, ...], ...]:
        """Adds the removed extra batch dimension to the actions before they are
        sent to the AsyncVectorEnv.
        """
        return add_extra_batch_dim(action, self.action_space_extra_dim)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return remove_extra_batch_dim_from_array(observation, self.observation_space)

    def reward(self, reward: Sequence[Sequence[T]]) -> List[T]:
        # TODO: There is no reward space atm, so how do we deal with this?
        assert len(reward) == self.n_workers, (reward, self.n_workers, self.group_lengths)
        return list(itertools.chain.from_iterable(reward))


def remove_extra_batch_dim_from_array(x: np.ndarray, space: gym.Space):
    if isinstance(space, spaces.Box):
        assert len(x.shape) >= 3, f"array {x} doesn't have an extra batch dim.."
        dims = space.shape
        x_dims = x.shape
        assert x_dims[2:] == space.shape[1:]
        return x.reshape([x_dims[0] * x_dims[1], *x_dims[2:]])

    raise NotImplementedError(f"TODO: support space {space}")

def add_extra_batch_dim(x: Union[np.ndarray, Tuple], space: gym.Space):
    """Adds an extra batch dimension to the item to fit the given space.
    """
    if isinstance(space, spaces.Box):
        assert isinstance(x, np.ndarray)
        assert len(space.shape) >= 3, "space doesn't have an extra batch dim.."
        return x.reshape(space.shape)

    if isinstance(space, spaces.Tuple):
        # Example:
        # space = Tuple(Tuple(Discrete(2), Discrete(2)), Tuple(Discrete(2), Discrete(2)))
        # (1, 0, 1, 0) -> ((1, 0), (1, 0))
        assert space.spaces and all(isinstance(sub_space, spaces.Tuple) for sub_space in space.spaces)
        assert isinstance(x, tuple)
        n_groups = len(space.spaces)
        max_group_length = int(len(x) / n_groups)
        # Assert that the groups except the last one have length max_group_length.
        for i, sub_space in enumerate(space.spaces):
            if i != len(space.spaces) - 1:
                assert len(sub_space.spaces) == max_group_length
            else:
                assert len(sub_space.spaces) <= max_group_length

        from operator import le, eq
        assert all(
            (le if i != (len(space.spaces) - 1) else eq)(
                len(sub_space.spaces), max_group_length
            ) for i, sub_space in enumerate(space.spaces)
        )
        return tuple(n_consecutive(x, n=max_group_length))


    raise NotImplementedError(f"TODO: support space {space}, x: {x}")


def remove_extra_batch_dim_from_space(space: gym.Space):
    if isinstance(space, spaces.Box):
        dims = space.shape
        assert len(space.shape) >= 3
        new_shape = tuple([dims[0] * dims[1], *dims[2:]])
        new_low = space.low.reshape(new_shape)
        new_high = space.high.reshape(new_shape)
        return spaces.Box(low=new_low, high=new_high, dtype=space.dtype)

    if isinstance(space, spaces.Tuple):
        assert len(space.spaces) and isinstance(space.spaces[0], spaces.Tuple)
        all_spaces: List[gym.Space] = []
        for sub_space in space.spaces:
            all_spaces.extend(sub_space.spaces)
        return spaces.Tuple(all_spaces)
    raise NotImplementedError(f"TODO: support space {space}")

