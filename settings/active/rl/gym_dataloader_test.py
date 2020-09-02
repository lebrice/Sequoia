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

from .gym_dataloader import GymDataLoader, GymDataset
from .gym_dataset_test import check_interaction_with_env

logger = get_logger(__file__)

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