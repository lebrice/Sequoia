
from typing import *

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor

from ..datasets.data_utils import FixChannels
from ..utils.logging_utils import get_logger, log_calls
from .environment_base import (EnvironmentBase, EnvironmentDataModule,
                               PassiveEnvironment)
from .rl_environments import GymEnvironment

logger = get_logger(__file__)


class DummyEnvironment(EnvironmentBase[int, int, float], IterableDataset):
    def __init__(self, start: int = 0):
        self.manager = mp.Manager()
        self.i: mp.Value[int] = self.manager.Value(int, start)

    @log_calls
    def __next__(self) -> int:
        val = self.i.value
        self.i.value += 1
        return val

    @log_calls
    def __iter__(self) -> Generator[int, int, None]:
        while True:
            action = yield next(self)
            if action is not None:
                logger.debug(f"Received an action of {action} while iterating..")
                self.reward = self.send(action)
    @log_calls
    def send(self, action: int) -> int:
        self.i.value += action
        return np.random.random()


def test_dummy_environment():
    # observations are Tensors, actions are ints (0, 1) and reward is a float.
    # bob: GymEnvironment[Tensor, int, float] = GymEnvironment(env=env)
    bob = DummyEnvironment()

    for i, x in enumerate(bob):
        assert x == i * 2

        bob.send(1)
        if i > 5:
            break


def test_gym_env_0_workers_batch_size_0():
    # env_factory = partial(gym.make, "CartPole-v0")
    env = gym.make("CartPole-v0")
    bob: GymEnvironment[Tensor, int, float] = GymEnvironment(env=env)
    data_module = EnvironmentDataModule(bob)
    loader = data_module.train_dataloader(batch_size=None, num_workers=0)

    for i, x in zip(range(5), loader):
        print(f"observation at step {i}: {x}")
        assert x.shape == (4,)
        assert isinstance(x, Tensor)
        y = loader.send(1)
        assert isinstance(y, float)
        logger.debug(f"reward: {y}, of type {type(y)}")


def test_passive_mnist_environment():

    dataset = MNIST("data", transform=Compose([ToTensor(), FixChannels()]))
    env: Iterable[Tuple[Tensor, Tensor]] = PassiveEnvironment(dataset)

    for x, y in env:
        print(x.shape, type(x), y)
        assert x.shape == (1, 3, 28, 28)
        x = x.permute(0, 2, 3, 1)
        assert y.item() == 5

        reward = env.send(4)
        assert reward is None
        # plt.imshow(x[0])
        # plt.title(f"y: {y[0]}")
        # plt.waitforbuttonpress(10)
        break
