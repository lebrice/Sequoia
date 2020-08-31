import multiprocessing as mp
from typing import Generator

from torch import Tensor

from utils.logging_utils import log_calls

from .active_dataloader import ActiveEnvironment
import numpy as np
from torchvision.datasets import MNIST
from common.transforms import Transforms, Compose

class DummyEnvironment(ActiveEnvironment[int, int, float]):
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
    
    def peek(self):
        return self.i.value

    def add(self, value: int) -> int:
        self.i.value += value
        return self.i.value
    
    def reset(self):
        self.i.value = 0

class ActiveMnistEnvironment(ActiveEnvironment[Tensor, Tensor, Tensor]):
    """ An Mnist environment which will keep showing the same class until a
    correct prediction is made, and then switch to another class.
    
    Which will keep giving the same class until the right prediction is made.
    """
    def __init__(self, start_class: int = 0, **kwargs):
        self.current_class: int = 0
        dataset = MNIST("data")
        super().__init__(dataset, batch_size=None, **kwargs)
        self.observation: Tensor = None
        self.reward: Tensor = None
        self.action: Tensor = None

    @log_calls
    def __next__(self) -> Tensor:
        for x, y in self.dataset:
            # keep iterating while the example isn't of the right type.
            if y == self.current_class:
                self.observation = x
                self.reward = y
                break

        print(f"next obs: {self.observation}, next reward = {self.reward}")
        return self.observation

    @log_calls
    def __iter__(self) -> Generator[Tensor, Tensor, None]:
        while True:
            action = yield next(self)
            if action is not None:
                logger.debug(f"Received an action of {action} while iterating..")
                self.reward = self.send(action)

    @log_calls
    def send(self, action: Tensor) -> Tensor:
        print(f"received action {action}, returning current label {self.reward}")
        self.action = action
        if action == self.current_class:
            print("Switching classes since the prediction was right!")
            self.current_class += 1
            self.current_class %= 10
        else:
            print("Prediction was wrong, staying on the same class.")
        return self.reward


def test_dummy_environment():
    # observations are Tensors, actions are ints (0, 1) and reward is a float.
    # bob: GymDataset[Tensor, int, float] = GymDataset(env=env)
    bob = DummyEnvironment()

    for i, x in enumerate(bob):
        assert x == i * 2

        bob.send(1)
        if i > 5:
            break



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
    _current_class = 0

    # Second loop, where we always predict the wrong label.
    for i, x in enumerate(env):
        print(f"x: {x}")
        y_pred = 1
        y_true = env.send(y_pred)
        assert y_true == 0

        if i > 2:
            break
    
    x = next(env)
    y_pred = 0
    y_true = env.send(y_pred)
    assert y_true == 0

    x = next(env)
    y_true = env.send(1)
    assert y_true == 1
