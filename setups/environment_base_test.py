
from typing import *

import gym
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import torch.multiprocessing as mp
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor

from ..datasets.data_utils import FixChannels
from ..utils.logging_utils import get_logger, log_calls
from .environment_base import (ActiveEnvironment, EnvironmentBase,
                               EnvironmentDataModule, PassiveEnvironment)
from .rl import GymEnvironment

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


class ActiveMnistEnvironment(ActiveEnvironment[Tensor, Tensor, Tensor]):
    """ An Mnist environment which will keep showing the same class until a
    correct prediction is made, and then switch to another class.
    
    which will keep giving the same class until the right prediction is made.

    Args:
        ActiveEnvironment ([type]): [description]
    """
    def __init__(self, start_class: int = 0, **kwargs):
        self.current_class: int = 0
        self.dataset = MNIST("data")
        super().__init__(data_source=self.dataset, batch_size=None, **kwargs)
        self.manager = mp.Manager()
        self.x: Tensor = None
        self.y: Tensor = None
        self.y_pred: Tensor = None

    @log_calls
    def __next__(self) -> int:
        while self.y != self.current_class:
            # keep iterating while the example isn't of the right type.
            self.x = super().__next__()
            self.y = super().send(None)

        print(f"next obs: {self.x}, next reward = {self.y}")
        return self.x

    @log_calls
    def __iter__(self) -> Generator[Tensor, Tensor, None]:
        while True:
            action = yield next(self)
            if action is not None:
                logger.debug(f"Received an action of {action} while iterating..")
                self.reward = self.send(action)

    @log_calls
    def send(self, action: Tensor) -> int:
        print(f"received action {action}, returning current label {self.y}")
        self.y_pred = action
        if action == self.current_class:
            print("Switching classes since the prediction was right!")
            self.current_class += 1
            self.current_class %= 10
        else:
            print("Prediction was wrong, staying on the same class.")
        return self.y


def test_active_mnist_environment():
    """Test the active mnist env, which will keep giving the same class until the right prediction is made.
    """
    env = ActiveMnistEnvironment()
    # So in this test, the env will only give samples of class 0, until a correct
    # prediction is made, then it will switch to giving samples of class 1, etc.

    # what the current class is (just for testing)
    _current_class = 0

    # first loop, where we always predict the right label.
    for i, x in enumerate(env):
        print(f"x: {x}")
        y_pred = i % 10
        print(f"Sending prediction of {y_pred}")
        y_true = env.send(y_pred)
        print(f"Received back {y_true}")
        
        assert y_pred == y_true
        if i == 9:
            break
    
    # current class should be 0 as last prediction was 9 and correct.

    # Second loop, where we always predict the wrong label.
    for i, x in enumerate(env):
        print(f"x: {x}")
        y_pred = 1
        y_true = env.send(y_pred)
        assert y_true == 0

        if i > 2:
            break
    
    x = next(env)
    y_true = env.send(0)
    assert y_true == 0

    x = next(env)
    y_true = env.send(1)
    assert y_true == 1
