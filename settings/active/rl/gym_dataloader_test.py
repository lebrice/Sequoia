from functools import wraps
from typing import Any, Callable, List, Tuple, Union

import gym
import numpy as np
import torch
from torch import Tensor

from conftest import xfail
from utils.logging_utils import get_logger

from .gym_dataset_test import check_interaction_with_env
from .gym_dataloader import GymDataset, GymDataLoader

logger = get_logger(__file__)
from utils import take

@xfail(reason="TODO: fix the weird batching behaviour..")
def test_batched_cartpole_state():
    batch_size = 10
    env: GymDataset[Tensor, int, float] = GymDataLoader("CartPole-v0", batch_size=batch_size, observe_pixels=False)
    obs_shape = (batch_size, 4)
    action = [1 for _ in range(batch_size)]
    for element in take(env.environments[0], 5):
        assert element.shape == obs_shape[1:]
    
    for batch in take(env, 5):
        assert batch.shape == obs_shape

    check_interaction_with_env(env, obs_shape=obs_shape, action=action)


@xfail(reason="TODO")
def test_cartpole_multiple_workers():
    assert False
