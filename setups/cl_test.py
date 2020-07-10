
from dataclasses import dataclass
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import torch.multiprocessing as mp
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset
from torchvision.transforms import Compose, ToTensor

from continuum.datasets import MNIST
from experiments.experiment import Experiment
from models.classifier import Classifier
from simple_parsing import ArgumentParser, mutable_field

from ..datasets.data_utils import FixChannels
from ..utils.logging_utils import get_logger, log_calls
from .cl import ClassIncrementalSetting, CLSetting
from .environment import (ActiveEnvironment, EnvironmentBase,
                          EnvironmentDataModule, PassiveEnvironment)
from .rl import GymEnvironment

logger = get_logger(__file__)


def test_class_incremental_mnist_setup():
    setup = ClassIncrementalMnist()
    
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


def test_class_incremental_mnist():
    """Test the active mnist env, which will keep giving the same class until the right prediction is made.
    """
    env = ClassIncrementalMnist()
    print(env)
    exit()
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
