
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
from .environment import (ActiveEnvironment, EnvironmentBase, PassiveEnvironment)
from .rl import GymEnvironment

logger = get_logger(__file__)

import itertools

def test_class_incremental_mnist_setup():
    env = ClassIncrementalSetting(dataset="mnist")
    env.prepare_data(data_dir="data")
    
    for i in range(5):
        env.current_task_id = i
        train_loader = env.train_dataloader()
        for x, y in itertools.islice(train_loader, 100):
            print(x.shape, type(x), y)
            assert x.shape == (1, 3, 28, 28)
            x = x.permute(0, 2, 3, 1)
            assert y.item() == 1
            print(type(train_loader))

            reward = train_loader.send(4)
            assert reward is None
            # plt.imshow(x[0])
            # plt.title(f"y: {y[0]}")
            # plt.waitforbuttonpress(10)
            break

