
import itertools
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
from datasets.data_utils import FixChannels
from experiments.experiment import Experiment
from models.classifier import Classifier
from simple_parsing import ArgumentParser, mutable_field
from utils.logging_utils import get_logger, log_calls

from setups.environment import (ActiveEnvironment, EnvironmentBase,
                           PassiveEnvironment)
from setups.rl import GymEnvironment
from setups.cl.base import ContinualSetting
from setups.cl.class_incremental import ClassIncrementalSetting

logger = get_logger(__file__)


def check_only_right_classes_present(env: ClassIncrementalSetting):
    """ Checks that only the classes within each task are present. """
    for i in range(env.nb_tasks):
        env.current_task_id = i
        train_loader = env.train_dataloader(batch_size=1)

        # Find out which classes are supposed to be within this task.
        if isinstance(env.increment, list):
            n_classes_in_task = env.increment[i]
            # # Find the starting index and end index in the list of classes.
            start_index = sum(env.increment[t] for t in range(i))
            end_index = start_index + env.increment[i]
        else:
            n_classes_in_task = env.increment
            start_index = env.increment * i
            end_index = env.increment * (i+1)

        # Use the custom class order if present, else the usual range 0-N
        classes_list = env.class_order or list(range(env.num_classes))
        classes_of_task = classes_list[start_index:end_index]

        for j, (x, y, t) in enumerate(itertools.islice(train_loader, 100)):
            print(i, j, y, t)
            assert y.item() in classes_of_task
            assert x.shape == (1, 3, 28, 28)
            x = x.permute(0, 2, 3, 1)[0]
            assert x.shape == (28, 28, 3)
            
            plt.imshow(x[0])
            
            reward = train_loader.send(4)
            assert reward is None


def test_class_incremental_mnist_setup():
    env = ClassIncrementalSetting(dataset="mnist", increment=2)
    env.prepare_data(data_dir="data")
    check_only_right_classes_present(env)

@pytest.mark.xfail(reason=(
    "Continuum actually re-labels the images to 0-10, regardless of the class "
    "order. The actual images are of the right type though. "
))
def test_class_incremental_mnist_setup_reversed_class_order():
    env = ClassIncrementalSetting(dataset="mnist", increment=2, class_order=list(reversed(range(10))))
    env.prepare_data(data_dir="data")
    check_only_right_classes_present(env)