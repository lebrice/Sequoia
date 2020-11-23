
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
from simple_parsing import ArgumentParser, mutable_field
from utils.logging_utils import get_logger, log_calls

from .task_incremental_setting import TaskIncrementalSetting

logger = get_logger(__file__)


def check_only_right_classes_present(env: TaskIncrementalSetting):
    """ Checks that only the classes within each task are present. """
    for i in range(env.nb_tasks):
        env.current_task_id = i
        batch_size = 5
        train_loader = env.train_dataloader(batch_size=batch_size)
        
        # # Find out which classes are supposed to be within this task.
        classes_of_task = env.current_task_classes(train=True)

        for j, (observations, rewards) in enumerate(itertools.islice(train_loader, 100)):
            x, t = observations
            y = rewards.y 
            print(i, j, y, t)
            assert all(y < env.n_classes_per_task)
            assert x.shape == (batch_size, 3, 28, 28)
            x = x.permute(0, 2, 3, 1)[0]
            assert x.shape == (28, 28, 3)
            
            reward = train_loader.send([4 for _ in range(batch_size)])
            assert reward is None


def test_class_incremental_mnist_setup():
    env = TaskIncrementalSetting(dataset="mnist", increment=2)
    env.prepare_data(data_dir="data")
    env.setup()
    check_only_right_classes_present(env)


@pytest.mark.xfail(reason=(
    "TODO: Continuum actually re-labels the images to 0-10, regardless of the "
    "class order. The actual images are ok though."
))
def test_class_incremental_mnist_setup_reversed_class_order():
    env = TaskIncrementalSetting(dataset="mnist", nb_tasks=5, class_order=list(reversed(range(10))))
    env.prepare_data(data_dir="data")
    env.setup()
    check_only_right_classes_present(env)

    # for i in range(env.nb_tasks):
    #     env.current_task_id = i
    #     train_loader = env.train_dataloader(batch_size=1)
        
    #     # # Find out which classes are supposed to be within this task.
    #     classes_of_task = env.current_task_classes()
    #     for j, (x, y, t) in enumerate(itertools.islice(train_loader, 100)):
    #         print(i, j, y, t)
    #         assert x.shape == (1, 3, 28, 28)
    #         x = x.permute(0, 2, 3, 1)[0]
    #         assert x.shape == (28, 28, 3)
            
    #         plt.imshow(x)
    #         plt.waitforbuttonpress()
            
    #         assert y.item() in classes_of_task
    #         reward = train_loader.send(4)
    #         assert reward is None


def test_class_incremental_mnist_setup_with_nb_tasks():
    env = TaskIncrementalSetting(dataset="mnist", nb_tasks=2)
    env.prepare_data(data_dir="data")
    env.setup()
    assert len(env.train_datasets) == 2
    assert len(env.val_datasets) == 2
    assert len(env.test_datasets) == 2
    check_only_right_classes_present(env)
