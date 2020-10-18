""" Mix of AsyncVectorEnv and SyncVectorEnv, with support for 'chunking' and for
where we have a series of environments on each worker.
"""
import math
import multiprocessing as mp
import itertools
from functools import partial
from typing import Any, Callable, Iterable, List, Sequence, Tuple, TypeVar, Union, Optional

import gym
import numpy as np
from gym import spaces
from gym.vector.sync_vector_env import SyncVectorEnv

from .async_vector_env import AsyncVectorEnv
from utils.utils import n_consecutive

from ..utils import space_with_new_shape

T = TypeVar("T")


class BatchEnv(gym.Env):
    """ Batched environment.

    Adds the following features, compared to using the vectorized environments
    from gym.vector:
    -   Chunking: Running more than one environment per worker. This is done by
        passing `SyncVectorEnv`s to the AsyncVectorEnv.
    -   Flexible batch size: Supports any number of environments, irrespective
        of the number of workers or of CPUs. The number of environments will be
        spread out as equally as possible between the workers.
      
        For example, if you want to have a batch_size of 17, and n_workers is 6,
        then the number of environments per worker will be: [3, 3, 3, 3, 3, 2].

        Internally, this works by creating up to two AsyncVectorEnvs, env_a and
        env_b. If the number of envs (batch_size) isn't a multiple of the number
        of workers, then we create the second AsyncVectorEnv (env_b).

        In the first environment (env_a), each env will contain
        ceil(n_envs / n_workers) each. If env_b is needed, then each of its envs
        will contain floor(n_envs / n_workers) environments.

    The observations/actions/rewards are reshaped to be (n_envs, *shape), i.e.
    they don't have an extra 'chunk' dimension.

    NOTE: The observation at index i in the batch isn't from the env env_fns[i].
    NOTE: In order to get this to work, I had to modify the `if done:` statement
    in the work to be `if done if isinstance(done, bool) else all(done)`.
    """
    def __init__(self,
                 env_fns,
                 n_workers: int = None,
                 **kwargs):
        assert env_fns, "need at least one env_fn."
        self.batch_size: int = len(env_fns)
        
        if n_workers is None:
            n_workers = mp.cpu_count()
        self.n_workers: int = n_workers

        if self.n_workers > self.batch_size:
            self.n_workers = self.batch_size

        # Divide the env_fns as evenly as possible between the workers.
        groups: List[List[Callable[[], gym.Env]]] = [[] for _ in range(self.n_workers)]
        for i, env_fn in enumerate(env_fns):
            groups[i % self.n_workers].append(env_fn)
        start_index_b = (i % self.n_workers) + 1

        self.n_a = sum(map(len, groups[:start_index_b]))
        self.n_b = sum(map(len, groups[start_index_b:]))

        # Create a SyncVectorEnv per group.
        chunk_env_fns: List[Callable[[], gym.Env]] = [
            partial(SyncVectorEnv, env_fns_group) for env_fns_group in groups
        ]

        env_a_fns = chunk_env_fns[:start_index_b]
        env_b_fns = chunk_env_fns[start_index_b:]
        
        # Create the AsyncVectorEnvs.
        self.chunk_a = math.ceil(self.batch_size / self.n_workers)
        self.env_a = AsyncVectorEnv(env_fns=env_a_fns, **kwargs)

        self.chunk_b = 0
        self.env_b: Optional[AsyncVectorEnv] = None

        if env_b_fns:
            self.chunk_b = math.floor(self.batch_size / self.n_workers)
            self.env_b = AsyncVectorEnv(env_fns=env_b_fns, **kwargs)

        self.observation_space: gym.Space
        self.action_space: gym.Space

        # Unbatch & join the observations/actions spaces.
        flat_obs_a = remove_extra_batch_dim_from_space(self.env_a.observation_space)
        flat_act_a = remove_extra_batch_dim_from_space(self.env_a.action_space)
        if self.env_b:
            flat_obs_b = remove_extra_batch_dim_from_space(self.env_b.observation_space)
            flat_act_b = remove_extra_batch_dim_from_space(self.env_b.action_space)
            
            self.observation_space = concat_spaces(flat_obs_a, flat_obs_b)
            self.action_space = concat_spaces(flat_act_a, flat_act_b)
        else:
            self.observation_space = flat_obs_a
            self.action_space = flat_act_a

        self.reward_range = self.env_a.reward_range

    def reset(self):
        obs_a = self.env_a.reset()
        if self.env_b:
            obs_b = self.env_b.reset()
            return self.concat_and_unchunk(obs_a, obs_b)
        return self.unchunk(obs_a)

    def step(self, action: Sequence):
        if self.env_b:
            flat_actions_a, flat_actions_b = action[:self.n_a], action[self.n_a:]
            actions_a = self.chunk(flat_actions_a, self.chunk_a)
            actions_b = self.chunk(flat_actions_b, self.chunk_b)

            obs_a, rew_a, done_a, info_a = self.env_a.step(actions_a)
            obs_b, rew_b, done_b, info_b = self.env_b.step(actions_b)

            observations = self.unchunk(obs_a, obs_b)
            rewards = self.unchunk(rew_a, rew_b)
            done = self.unchunk(done_a, done_b)
            info = self.unchunk(info_a, info_b)
            return observations, rewards, done, info

        action = self.chunk(action, self.chunk_a)
        observations, rewards, done, info = self.env_a.step(action)
        observations = self.unchunk(observations)
        rewards = self.unchunk(rewards)
        done = self.unchunk(done)
        info = self.unchunk(info)
        return observations, rewards, done, info

    def seed(self, seeds: Sequence[int]=None):
        if seeds is None:
            seeds = [None for _ in range(self.batch_size)]
        if isinstance(seeds, int):
            seeds = [seeds + i for i in range(self.batch_size)]
        assert len(seeds) == self.batch_size

        seeds_a = self.chunk(seeds[:self.n_a], self.chunk_a)
        seeds_b = self.chunk(seeds[self.n_a:], self.chunk_b)
        self.env_a.seed(seeds_a)
        if self.env_b:
            self.env_b.seed(seeds_b)       
        self.env_a.render

    def close(self):
        self.env_a.close()
        if self.env_b:
            self.env_b.close()

    def unchunk(self, *values: Sequence[Sequence[T]]) -> Sequence[T]:
        """ Combine 'chunked' results coming from the envs into a single
        batch.
        """
        all_values: List[T] = []
        for sequence in values:
            all_values.extend(itertools.chain.from_iterable(sequence))
        if isinstance(values[0], np.ndarray):
            return np.array(all_values)
        return all_values

    def chunk(self, values: Sequence[T], chunk_length: int) -> Sequence[Sequence[T]]:
        """ Add the 'chunk'/second batch dimension to the list of items. """
        groups = list(n_consecutive(values, chunk_length))
        if isinstance(values, np.ndarray):
            groups = np.array(groups)
        return groups


def concat_spaces(space_a: gym.Space, space_b: gym.Space) -> gym.Space:
    """ Concatenate two gym spaces of the same types. """
    if type(space_a) != type(space_b):
        raise RuntimeError(f"Can only concat spaces of the same type: {space_a} {space_b}")
    
    if isinstance(space_a, spaces.Box):
        assert space_a.shape[1:] == space_b.shape[1:]
        new_low = np.concatenate([space_a.low, space_b.low])
        new_high = np.concatenate([space_a.high, space_b.high])
        return spaces.Box(low=new_low, high=new_high, dtype=space_a.dtype)

    if isinstance(space_a, spaces.Tuple):
        new_spaces = space_a.spaces + space_b.spaces
        return spaces.Tuple(new_spaces)
    
    if isinstance(space_a, spaces.Dict):
        new_spaces = {}
        for key in space_a.spaces:
            value_a = space_a[key]
            value_b = space_b[key]
            new_spaces[key] = concat_spaces(value_a, value_b)
        return spaces.Dict(new_spaces)
    
    raise NotImplementedError(f"Unsupported spaces {space_a} {space_b}")


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

    if isinstance(space, spaces.Dict):
        return spaces.Dict({
            k: remove_extra_batch_dim_from_space(sub_space)
            for k, sub_space in space.spaces.items()
        })

    raise NotImplementedError(f"Unsupported space {space}")
