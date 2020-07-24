from .rl import GymEnvironment
from torch import Tensor
import gym
import numpy as np
from utils.logging_utils import get_logger


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
