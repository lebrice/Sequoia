"""
Tests that check that combining wrappers works fine in combination.
"""

from typing import Callable, Union

import gym
import pytest
import torch

from .make_env import make_batched_env


@pytest.mark.parametrize("env_name", ["CartPole-v0"])
@pytest.mark.parametrize("batch_size", [1, 5, 10, 32])
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


@pytest.mark.parametrize("env_name", ["CartPole-v0"])
@pytest.mark.parametrize("batch_size", [1, 5, 10, 32])
def test_make_env_with_wrapper(env_name: str, batch_size: int):
    env = make_batched_env(
        base_env=env_name,
        batch_size=batch_size,
        wrappers=[PixelObservationWrapper],
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
from sequoia.common.gym_wrappers.batch_env import AsyncVectorEnv


@pytest.mark.xfail(reason=f"TODO: Haven't added the env_method or env_attribute or set_attr methods on the BatchedVectorEnv.")
@pytest.mark.parametrize("env_name", ["CartPole-v0"])
@pytest.mark.parametrize("batch_size", [1, 5, 10, 32])
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

