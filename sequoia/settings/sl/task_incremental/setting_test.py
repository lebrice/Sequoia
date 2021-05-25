
import itertools
import math
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
from sequoia.common.config import Config
from sequoia.utils.logging_utils import get_logger, log_calls
from sequoia.settings.assumptions.incremental_test import OtherDummyMethod
from .setting import TaskIncrementalSLSetting

logger = get_logger(__file__)


def check_only_right_classes_present(setting: TaskIncrementalSLSetting):
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
            y_in_task_classes = [y_i in task_classes for y_i in y.tolist()]
            assert all(y_in_task_classes)
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
    setting = TaskIncrementalSLSetting(
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
    setting = TaskIncrementalSLSetting(
        dataset="mnist",
        nb_tasks=5,
        class_order=list(reversed(range(10))),
        # num_workers=0,
    )
    setting.prepare_data(data_dir="data")
    setting.setup()
    check_only_right_classes_present(setting)


def test_class_incremental_mnist_setup_with_nb_tasks():
    setting = TaskIncrementalSLSetting(
        dataset="mnist",
        nb_tasks=2,
        num_workers=0,
    )
    assert setting.increment == 5
    setting.prepare_data(data_dir="data")
    setting.setup()
    assert len(setting.train_datasets) == 2
    assert len(setting.val_datasets) == 2
    assert len(setting.test_datasets) == 2
    check_only_right_classes_present(setting)


def test_action_space_always_matches_obs_batch_size(config: Config):
    """ Make sure that the batch size in the observations always matches the action
    space provided to the `get_actions` method.

    ALSO:
    - Make sure that we get asked for actions for all the observations in the test set,
      even when there is a shorter last batch.
    - The total number of observations match the dataset size.
    """
    nb_tasks = 5
    batch_size = 128

    # HUH why are we doing this here?
    setting = TaskIncrementalSLSetting(
        dataset="mnist",
        nb_tasks=nb_tasks,
        batch_size=batch_size,
        num_workers=4,
        monitor_training_performance=True,
    )

    # 10_000 examples in the test dataset of mnist.
    total_samples = len(setting.test_dataloader().dataset)

    method = OtherDummyMethod()
    _ = setting.apply(method, config=config)

    # Multiply by nb_tasks because the test loop is ran after each training task.
    assert sum(method.batch_sizes) == total_samples * nb_tasks
    assert len(method.batch_sizes) == math.ceil(total_samples / batch_size) * nb_tasks
    assert set(method.batch_sizes) == {batch_size, total_samples % batch_size}
