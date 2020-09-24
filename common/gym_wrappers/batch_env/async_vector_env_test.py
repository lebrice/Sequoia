from functools import partial
from operator import attrgetter

import gym
import pytest
from gym.envs.classic_control import CartPoleEnv

from .async_vector_env import AsyncVectorEnv


@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_apply(batch_size: int):
    env_fns = [partial(gym.make, "CartPole-v0") for _ in range(batch_size)]
    env: AsyncVectorEnv
    with AsyncVectorEnv(env_fns=env_fns) as env:
        results = env.apply(attrgetter("length"))
        # Get the pole lengths.
        assert results == [0.5 for i in range(batch_size)]



@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_getitem_as_proxy(batch_size: int):
    env_fns = [partial(gym.make, "CartPole-v0") for _ in range(batch_size)]
    env: AsyncVectorEnv[CartPoleEnv]
    with AsyncVectorEnv(env_fns=env_fns) as env:
        env: AsyncVectorEnv[CartPoleEnv] = env
        # Set the pole length to 2.0 but only in the first environment.
        env[0].length = 2.0
        assert env[0].length == 2.0

        lengths = env[:].length
        # Get the pole lengths, check that the first env has a different value!
        assert lengths == [2.0 if i == 0 else 0.5 for i in range(batch_size)]


def test_getitem_with_slice():
    batch_size: int = 4
    env_fns = [partial(gym.make, "CartPole-v0") for _ in range(batch_size)]
    env: AsyncVectorEnv[CartPoleEnv]
    with AsyncVectorEnv(env_fns=env_fns) as env:
        env: AsyncVectorEnv[CartPoleEnv] = env
        # Set the pole length of the first 2 environments to 2.0.
        env[0:2].length = 2.0
        assert env[0].length == 2.0
        assert env[1].length == 2.0
        assert env[2].length == 0.5
        assert env[3].length == 0.5

        assert env[1:].length == [2.0, 0.5, 0.5]





@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_getattr_gets_it_from_envs(batch_size: int):
    env_fns = [partial(gym.make, "CartPole-v0") for _ in range(batch_size)]
    env: AsyncVectorEnv[CartPoleEnv]
    with AsyncVectorEnv(env_fns=env_fns) as env:
        env: AsyncVectorEnv[CartPoleEnv] = env
        # Set the pole length to 2.0 but only in the first environment.
        env[0].length = 2.0

        # Since the env doesn't have the attribute, it will try to get it from the envs.
        lengths = env.length
        # Get the pole lengths, check that the first env has a different value!
        assert lengths == [2.0 if i == 0 else 0.5 for i in range(batch_size)]
