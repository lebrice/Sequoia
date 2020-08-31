from functools import wraps
from typing import Any, Callable, List, Tuple, Union, Any

import gym
import numpy as np
import torch
from torch import Tensor

from conftest import xfail
from settings.base import EnvironmentBase
from utils.logging_utils import get_logger

from .gym_dataloader import GymDataLoader
from .gym_dataset import GymDataset

logger = get_logger(__file__)


def check_interaction_with_env(env: Union[GymDataset, GymDataLoader],
                               obs_shape: Tuple[int, ...],
                               action: Any=1,
                               n_batches: int=5):
    """Get a few batches from the env and make sure they have the desired shape.
    Also sends the given action at each step.
    """
    for i, x in zip(range(n_batches), env):
        assert isinstance(x, (Tensor, np.ndarray))
        assert x.shape == obs_shape
        reward = env.send(action)
        assert isinstance(reward, (float, Tensor))
    env.close()


def test_wrap_cartpole_state():
    env = gym.make("CartPole-v0")
    bob: GymDataset[Tensor, int, float] = GymDataset(env=env, observe_pixels=False)
    check_interaction_with_env(bob, obs_shape=(4,), action=1)


def test_wrap_cartpole_pixels():
    env = gym.make("CartPole-v0")
    bob: GymDataset[Tensor, int, float] = GymDataset(env=env, observe_pixels=True)
    check_interaction_with_env(bob, obs_shape=(400, 600, 3), action=1)


def test_cartpole_state():
    env: GymDataset[Tensor, int, float] = GymDataset("CartPole-v0", observe_pixels=False)
    check_interaction_with_env(env, obs_shape=(4,), action=1)


def test_cartpole_pixels():
    bob: GymDataset[Tensor, int, float] = GymDataset("CartPole-v0", observe_pixels=True)
    check_interaction_with_env(bob, obs_shape=(400, 600, 3))
