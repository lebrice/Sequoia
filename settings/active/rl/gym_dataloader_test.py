from functools import partial, reduce, wraps
from typing import (Any, Callable, Dict, Iterable, List, Optional, Tuple, Type,
                    TypeVar, Union)

import gym
import numpy as np
import pytest
import torch
from common.gym_wrappers import (AsyncVectorEnv, ConvertToFromTensors,
                                 EnvDataset, PixelStateWrapper)
from common.transforms import ChannelsFirstIfNeeded
from conftest import DummyEnvironment, xfail
from gym.envs.classic_control import CartPoleEnv, PendulumEnv
from common.gym_wrappers import TransformObservation
from torch import Tensor
from utils import take
from utils.logging_utils import get_logger

from .gym_dataloader import GymDataLoader
from .make_env import default_wrappers_for_env

logger = get_logger(__file__)



def test_max_steps_is_respected():
    epochs = 3
    max_steps = 5
    env: GymDataLoader = GymDataLoader(
        env="CartPole-v0",
        # pre_batch_wrappers=wrappers,
        max_steps=max_steps,
    )
    for epoch in range(epochs):
        for i, batch in enumerate(env):
            assert i < max_steps, f"Max steps should have been respected: {i}"
            env.send(env.action_space.sample())
        assert i == max_steps - 1
    assert epoch == epochs - 1
    env.close()



@pytest.mark.parametrize("env_name", ["CartPole-v0"])
@pytest.mark.parametrize("batch_size", [1, 2, 5, 10])
def test_batched_cartpole_state(env_name: str, batch_size: int):
    env: GymDataLoader = GymDataLoader(
        env_name,
        batch_size=batch_size,
    )
    with gym.make(env_name) as temp_env:
        state_shape = temp_env.observation_space.shape
        action_shape = temp_env.action_space.shape

    state_shape = (batch_size, *state_shape)
    action_shape = (batch_size, *action_shape)
    reward_shape = (batch_size,)

    state = env.reset()
    assert state.shape == state_shape
    env.seed(123)
    i = 0
    for obs_batch, done, info in take(env, 5):
        assert obs_batch.shape == state_shape

        random_actions = env.action_space.sample()
        assert torch.as_tensor(random_actions).shape == action_shape
        assert temp_env.action_space.contains(random_actions[0])

        reward = env.send(random_actions)
        assert reward.shape == reward_shape
        i += 1
    assert i == 5

@pytest.mark.parametrize("env_name", ["CartPole-v0"])
@pytest.mark.parametrize("batch_size", [1, 2, 5, 10])
def test_batched_cartpole_pixels(env_name: str, batch_size: int):
    wrappers = default_wrappers_for_env[env_name] + [PixelStateWrapper]
    env: GymDataLoader = GymDataLoader(
        env_name,
        pre_batch_wrappers=wrappers,
        batch_size=batch_size,
    )
    with gym.make(env_name) as temp_env:
        for wrapper in wrappers:
            temp_env = wrapper(temp_env)
        state_shape = temp_env.observation_space.shape
        action_shape = temp_env.action_space.shape
    state_shape = (batch_size, *state_shape)
    action_shape = (batch_size, *action_shape)
    reward_shape = (batch_size,)

    state = env.reset()
    assert state.shape == state_shape
    env.seed(123)
    i = 0
    for obs_batch, done, info in take(env, 5):
        assert obs_batch.shape == state_shape

        random_actions = env.action_space.sample()
        assert torch.as_tensor(random_actions).shape == action_shape
        assert temp_env.action_space.contains(random_actions[0])

        reward = env.send(random_actions)
        assert reward.shape == reward_shape
        i += 1
    assert i == 5


@pytest.mark.parametrize("env_name", ["CartPole-v0"])
@pytest.mark.parametrize("batch_size", [1, 2, 5, 10])
def test_channels_first_wrapper(env_name: str, batch_size: int):
    wrappers = default_wrappers_for_env[env_name] + [
        PixelStateWrapper,
        partial(TransformObservation, f=ChannelsFirstIfNeeded())
    ]
    env: GymDataLoader = GymDataLoader(
        env_name,
        pre_batch_wrappers=wrappers,
        batch_size=batch_size,
    )
    with gym.make(env_name) as temp_env:
        for wrapper in wrappers:
            temp_env = wrapper(temp_env)
        state_shape = temp_env.observation_space.shape
        action_shape = temp_env.action_space.shape
    state_shape = (batch_size, *state_shape)
    action_shape = (batch_size, *action_shape)
    reward_shape = (batch_size,)

    state = env.reset()
    assert state.shape == state_shape
    env.seed(123)
    i = 0
    for obs_batch, done, info in take(env, 5):
        assert obs_batch.shape == state_shape

        random_actions = env.action_space.sample()
        assert torch.as_tensor(random_actions).shape == action_shape
        assert temp_env.action_space.contains(random_actions[0])

        reward = env.send(random_actions)
        assert reward.shape == reward_shape
        i += 1
    assert i == 5



@pytest.mark.parametrize("env_name", ["CartPole-v0"])
@pytest.mark.parametrize("batch_size", [1, 2, 5, 10])
def test_channels_post_batch_wrapper(env_name: str, batch_size: int):
    pre_batch_wrappers = default_wrappers_for_env[env_name] + [
        PixelStateWrapper,
    ]
    post_batch_wrappers = [
        partial(TransformObservation, f=ChannelsFirstIfNeeded())
    ]
    env: GymDataLoader = GymDataLoader(
        env_name,
        pre_batch_wrappers=pre_batch_wrappers,
        post_batch_wrappers=post_batch_wrappers,
        batch_size=batch_size,
    )
    with gym.make(env_name) as temp_env:
        # Apply the pre batch wrappers
        for wrapper in pre_batch_wrappers:
            temp_env = wrapper(temp_env)
        # Apply the post batch wrappers
        for wrapper in post_batch_wrappers:
            temp_env = wrapper(temp_env)

        state_shape = temp_env.observation_space.shape
        action_shape = temp_env.action_space.shape
    state_shape = (batch_size, *state_shape)
    action_shape = (batch_size, *action_shape)
    reward_shape = (batch_size,)

    state = env.reset()
    assert state.shape == state_shape
    env.seed(123)
    i = 0
    for obs_batch, done, info in take(env, 5):
        assert obs_batch.shape == state_shape

        random_actions = env.action_space.sample()
        assert torch.as_tensor(random_actions).shape == action_shape
        assert temp_env.action_space.contains(random_actions[0])

        reward = env.send(random_actions)
        assert reward.shape == reward_shape
        i += 1
    assert i == 5

