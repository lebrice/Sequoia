"""
Tests that check that combining wrappers works fine in combination.
"""

from typing import Callable, Union

import gym
import pytest
import torch
from sequoia.conftest import slow_param
from .make_env import make_batched_env
from gym.vector import SyncVectorEnv, AsyncVectorEnv


@pytest.mark.parametrize("env_name", ["CartPole-v0"])
@pytest.mark.parametrize("batch_size", [1, 5, slow_param(10)])
def test_make_batched_env(env_name: str, batch_size: int):
    env = make_batched_env(base_env=env_name, batch_size=batch_size)
    start_state = env.reset()
    assert start_state.shape == (batch_size, 4)

    for i in range(10):
        action = env.action_space.sample()
        assert torch.as_tensor(action).shape == (batch_size,)
        obs, reward, done, info = env.step(action)
        assert obs.shape == (batch_size, 4)
        assert reward.shape == (batch_size,)


@pytest.mark.xfail(
    reason="Not sure that the 'id' function gives an 'absolute' memory adress, or if "
    "the address is process-relative, in which case it might be an explanation as to "
    "why these tests don't work."
)
@pytest.mark.parametrize("env_name", ["CartPole-v0"])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("num_workers", [0, 4])
def test_make_batched_env_envs_have_distinct_ids(
    env_name: str, batch_size: int, num_workers: int
):
    # NOTE: We get a SyncVectorEnv if num_workers == 0, else we get an AsyncVectorEnv if
    # num_workers == batch_size, else we get a BatchVectorEnv.
    from gym.wrappers import TimeLimit

    def base_env_fn():
        env = gym.make(env_name)
        return TimeLimit(env, max_episode_steps=10)

    env: Union[SyncVectorEnv, AsyncVectorEnv] = make_batched_env(
        base_env=base_env_fn, batch_size=batch_size, num_workers=num_workers
    )
    if isinstance(env, SyncVectorEnv):
        envs = env.envs
        # Assert that the wrappers are distinct objects
        assert len(set(id(env) for env in envs)) == batch_size
        # Assert that the unwrapped envs are distinct objects
        assert len(set(id(env.unwrapped) for env in envs)) == batch_size
    else:
        assert isinstance(env, AsyncVectorEnv)
        ids = env.apply(id)
        assert len(set(ids)) == batch_size
        unwrapped_ids = env.apply(get_unwrapped_id)
        assert len(set(unwrapped_ids)) == batch_size


def get_unwrapped_id(env):
    return id(env.unwrapped)


@pytest.mark.parametrize("env_name", ["CartPole-v0"])
@pytest.mark.parametrize("batch_size", [1, 5, slow_param(10)])
def test_make_env_with_wrapper(env_name: str, batch_size: int):
    env = make_batched_env(
        base_env=env_name, batch_size=batch_size, wrappers=[PixelObservationWrapper],
    )
    start_state = env.reset()
    expected_state_shape = (batch_size, 400, 600, 3)
    assert start_state.shape == expected_state_shape

    for i in range(10):
        action = env.action_space.sample()
        assert torch.as_tensor(action).shape == (batch_size,)
        obs, reward, done, info = env.step(action)
        assert obs.shape == expected_state_shape
        assert reward.shape == (batch_size,)


from sequoia.common.gym_wrappers import PixelObservationWrapper, MultiTaskEnvironment
from gym.vector import AsyncVectorEnv


@pytest.mark.parametrize("env_name", ["CartPole-v0"])
@pytest.mark.parametrize("batch_size", [1, 5, slow_param(10)])
def test_make_env_with_wrapper_and_kwargs(env_name: str, batch_size: int):
    env = make_batched_env(
        base_env=env_name,
        batch_size=batch_size,
        wrappers=[
            PixelObservationWrapper,
            (MultiTaskEnvironment, dict(task_schedule={0: dict(length=2.0)})),
        ],
        # For now, setting the number of workers to the batch size, just so we
        # get an AsyncVectorEnv rather than the BatchedVectorEnv (so the remote_getattr works).
        num_workers=batch_size,
    )
    AsyncVectorEnv.allow_remote_getattr = True

    start_state = env.reset()
    expected_state_shape = (batch_size, 400, 600, 3)
    assert start_state.shape == expected_state_shape

    for i in range(10):
        action = env.action_space.sample()
        assert torch.as_tensor(action).shape == (batch_size,)

        assert env.length == [2.0 for i in range(batch_size)]

        obs, reward, done, info = env.step(action)
        assert obs.shape == expected_state_shape
        assert reward.shape == (batch_size,)

