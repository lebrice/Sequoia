
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
from sequoia.utils.logging_utils import get_logger, log_calls

from .task_incremental_setting import TaskIncrementalSetting

logger = get_logger(__file__)


def check_only_right_classes_present(setting: TaskIncrementalSetting):
    """ Checks that only the classes within each task are present. """
    for i in range(setting.nb_tasks):
        setting.current_task_id = i
        batch_size = 5
        train_loader = setting.train_dataloader(batch_size=batch_size)
        
        # get the classes in the current task:
        task_classes = setting.task_classes(i, train=True)
        
        for j, (observations, rewards) in enumerate(itertools.islice(train_loader, 100)):
            x = observations.x
            t = observations.task_labels
            y = rewards.y
            print(i, j, y, t)
            assert all(y_i in task_classes for y_i in y.tolist())
            assert x.shape == (batch_size, 3, 28, 28)
            x = x.permute(0, 2, 3, 1)[0]
            assert x.shape == (28, 28, 3)
            
            reward = train_loader.send([4 for _ in range(batch_size)])
            if setting.monitor_training_performance:
                assert reward is None
            else:
                assert (reward.y == rewards.y).all()

        train_loader.close()


def test_task_incremental_mnist_setup():
    setting = TaskIncrementalSetting(
        dataset="mnist",
        increment=2,
        # BUG: When num_workers > 0, some of the tests hang, but only when running *all* the tests!
        # num_workers=0,
    )
    setting.prepare_data(data_dir="data")
    setting.setup()
    check_only_right_classes_present(setting)


@pytest.mark.xfail(reason=(
    "TODO: Continuum actually re-labels the images to 0-10, regardless of the "
    "class order. The actual images are ok though."
))
def test_class_incremental_mnist_setup_reversed_class_order():
    setting = TaskIncrementalSetting(
        dataset="mnist",
        nb_tasks=5,
        class_order=list(reversed(range(10))),
        # num_workers=0,
    )
    setting.prepare_data(data_dir="data")
    setting.setup()
    check_only_right_classes_present(setting)


def test_class_incremental_mnist_setup_with_nb_tasks():
    setting = TaskIncrementalSetting(
        dataset="mnist",
        nb_tasks=2,
        num_workers=0,
    )
    setting.prepare_data(data_dir="data")
    setting.setup()
    assert len(setting.train_datasets) == 2
    assert len(setting.val_datasets) == 2
    assert len(setting.test_datasets) == 2
    check_only_right_classes_present(setting)
