from functools import wraps
from typing import Any, Callable, List, Optional, Tuple, Union

import gym
import numpy as np
import pytest
import torch
from torch import Tensor

from conftest import xfail
from utils import take
from utils.logging_utils import get_logger

from .gym_dataloader import GymDataLoader
from common.gym_wrappers import EnvDataset
from conftest import DummyEnvironment


logger = get_logger(__file__)


from common.gym_wrappers import BatchEnv, EnvDataset, MultiTaskEnvironment, PixelStateWrapper

def make_env_factory(env_name: str = "CartPole-v0", batch_size: int = 10):
    def env_factory():
        # TODO: Figure out the right ordering to use for the wrappers.
        # env = gym.make(env_name)
        # env = PixelStateWrapper(env)
        env = BatchEnv(env_name, batch_size=batch_size)
        env = EnvDataset(env)
        # env = MultiTaskEnvironment(env)
        return env
    return env_factory


@pytest.mark.parametrize("env_name", ["CartPole-v0"])
@pytest.mark.parametrize("batch_size", [1, 5, 10, 32])
def test_all_wrappers_work(env_name: str, batch_size: int):
    env_factory = make_env_factory(env_name=env_name, batch_size=batch_size)
    env: Union[BatchEnv, EnvDataset] = env_factory()
    start_state = env.reset()
    assert start_state.shape == (batch_size, 4)

    for i in range(10):
        action = env.random_actions()
        assert action.shape == (batch_size,)
        obs, reward, done, info = env.step(action)
        assert obs.shape == (batch_size, 4)
        assert reward.shape == (1,)

# TODO: Finish updating the rest!


@pytest.mark.parametrize("batch_size", [1, 2, 5, 10])
def test_batched_cartpole_state(batch_size: int):

    env: GymDataset[Tensor, int, float] = GymDataLoader(
        "CartPole-v0",
        batch_size=batch_size,
        observe_pixels=False,
    )
    obs_shape = (batch_size, 4)
    reward_shape = (batch_size,)

    env.reset()
    for obs_batch in take(env, 5):
        assert obs_batch.shape == obs_shape

        random_actions = env.random_actions()
        reward = env.send(random_actions)
        assert reward.shape == reward_shape

    check_interaction_with_env(
        env,
        obs_shape=obs_shape,
        action=random_actions,
        reward_shape=reward_shape,
    )


@pytest.mark.parametrize("num_workers", [0, 1, 2, 5, 10, 24])
def test_cartpole_multiple_workers(num_workers: Optional[int]):
    batch_size = num_workers or 32
    env: GymDataset[Tensor, int, float] = GymDataLoader(
        "CartPole-v0",
        batch_size=batch_size,
        observe_pixels=False,
        num_workers=num_workers,
    )
    obs_shape = (batch_size, 4)
    reward_shape = (batch_size,)
    reward_shape = (batch_size,)

    env.reset()
    for obs_batch in take(env, 5):
        assert obs_batch.shape == obs_shape

        random_actions = env.random_actions()
        reward = env.send(random_actions)
        assert reward.shape == reward_shape


def test_raise_error_when_missing_action():
    env = GymDataLoader(
        "CartPole-v0",
        batch_size=10,
        observe_pixels=False,
        random_actions_when_missing=True,
    )
    env.reset()
    for obs_batch in take(env, 5):
        # doesn't complain when you give back no action, it uses a random one.
        pass
        # env.send(env.random_actions())

    env = GymDataLoader(
        "CartPole-v0",
        batch_size=10,
        observe_pixels=False,
        random_actions_when_missing=False,
    )
    env.reset()
    for obs_batch in take(env, 5):
        # doesn't complain when you give back and action.
        env.send(env.random_actions())

    env = GymDataLoader(
        "CartPole-v0",
        batch_size=10,
        observe_pixels=False,
        random_actions_when_missing=False,
    )
    env.reset()
    with pytest.raises(RuntimeError):
        for obs_batch in take(env, 5):
            # raises an error after the first iteration, as it didn't receive an action. 
            pass

    