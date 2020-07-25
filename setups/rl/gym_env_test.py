from functools import wraps
from typing import Any, Callable, List

import gym
import numpy as np
import torch
from torch import Tensor

from utils.logging_utils import get_logger

from .batched_env import BatchEnvironments
from .gym_env import BatchedGymEnvironment, GymEnvironment

logger = get_logger(__file__)

def test_gym_env_0_workers_batch_size_0():
    # env_factory = partial(gym.make, "CartPole-v0")
    env = gym.make("CartPole-v0")
    bob: GymEnvironment[Tensor, int, float] = GymEnvironment(env=env)

    for i, x in zip(range(5), bob):
        print(f"observation at step {i}: {x}")
        assert x.shape == (4,)
        assert isinstance(x, (Tensor, np.ndarray))
        y = bob.send(1)
        assert isinstance(y, float)
        logger.debug(f"reward: {y}, of type {type(y)}")

    bob.close()


def test_batched_gym_env():
    batch_size = 2
    envs = [
        GymEnvironment("CartPole-v0") for i in range(batch_size)
    ]
    bob = BatchedGymEnvironment(*envs)
    
    for i, x in zip(range(5), bob):
        x = torch.as_tensor(x)
        print(f"observation at step {i}: {x.shape}")
        assert x.shape == (2, 4)
        assert isinstance(x, (Tensor))
        print(bob.action_space)
        bob.send(bob.random_actions())

    bob.close()
