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
        action = env.random_actions()
        assert torch.as_tensor(action).shape == (batch_size,)
        obs, reward, done, info = env.step(action)
        assert obs.shape == (batch_size, 4)
        assert reward.shape == (batch_size,)


@pytest.mark.parametrize("env_name", ["CartPole-v0"])
@pytest.mark.parametrize("batch_size", [
    1,
    5,
    10,
    32,
])
def test_make_env_with_wrapper(env_name: str, batch_size: int):
    env = make_batched_env(base_env=env_name, batch_size=batch_size)
    start_state = env.reset()
    expected_state_shape = (batch_size, 400, 600, 3)
    assert start_state.shape == expected_state_shape

    for i in range(10):
        action = env.random_actions()
        assert torch.as_tensor(action).shape == (batch_size,)
        obs, reward, done, info = env.step(action)
        assert False, done
        assert obs.shape == expected_state_shape
        assert reward.shape == (batch_size,)



@pytest.mark.parametrize("env_name", ["CartPole-v0"])
@pytest.mark.parametrize("batch_size", [
    1,
    5,
    10,
    32,
])
def test_make_env_with_wrapper_and_kwargs(env_name: str, batch_size: int):
    env = make_batched_env(
        base_env=env_name,
        batch_size=batch_size,
    )
    start_state = env.reset()
    expected_state_shape = (batch_size, 400, 600, 3)
    assert start_state.shape == expected_state_shape

    for i in range(10):
        action = env.random_actions()
        assert torch.as_tensor(action).shape == (batch_size,)
        obs, reward, done, info = env.step(action)
        assert False, done
        assert obs.shape == expected_state_shape
        assert reward.shape == (batch_size,)

