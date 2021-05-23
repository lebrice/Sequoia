import pytest
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, TensorDataset
from avalanche.benchmarks import nc_benchmark

from sequoia.common.config import Config
from sequoia.settings.sl import ClassIncrementalSetting, TaskIncrementalSetting
from sequoia.client import SettingProxy


@pytest.fixture(scope="session")
def fast_scenario(use_task_labels=False, shuffle=True):
    """ Copied directly from Avalanche in "tests/unit_tests_utils.py".

    Not used anywhere atm, but could be used as inspiration for writing quicker tests
    in Sequoia.
    """
    n_samples_per_class = 100
    dataset = make_classification(
        n_samples=10 * n_samples_per_class,
        n_classes=10,
        n_features=6,
        n_informative=6,
        n_redundant=0,
    )

    X = torch.from_numpy(dataset[0]).float()
    y = torch.from_numpy(dataset[1]).long()

    train_X, test_X, train_y, test_y = train_test_split(
        X, y, train_size=0.6, shuffle=True, stratify=y
    )

    train_dataset = TensorDataset(train_X, train_y)
    test_dataset = TensorDataset(test_X, test_y)
    my_nc_benchmark = nc_benchmark(
        train_dataset, test_dataset, 5, task_labels=use_task_labels, shuffle=shuffle
    )
    return my_nc_benchmark


@pytest.fixture(scope="session")
def short_task_incremental_setting(config: Config):
    setting = TaskIncrementalSetting(
        dataset="mnist", nb_tasks=5, monitor_training_performance=True,
    )
    setting.config = config
    setting.prepare_data()
    setting.setup("train")

    # Testing this out: Shortening the train datasets:
    setting.train_datasets = [
        Subset(task_dataset, list(range(100)))
        for task_dataset in setting.train_datasets
    ]
    setting.val_datasets = [
        Subset(task_dataset, list(range(100))) for task_dataset in setting.val_datasets
    ]
    setting.test_datasets = [
        Subset(task_dataset, list(range(100))) for task_dataset in setting.test_datasets
    ]
    return setting


@pytest.fixture(scope="session")
def short_class_incremental_setting(config: Config):
    setting = ClassIncrementalSetting(
        dataset="mnist", nb_tasks=5, monitor_training_performance=True,
    )
    setting.config = config
    setting.prepare_data()
    setting.setup("train")

    # Testing this out: Shortening the train datasets:
    setting.train_datasets = [
        Subset(task_dataset, list(range(100)))
        for task_dataset in setting.train_datasets
    ]
    setting.val_datasets = [
        Subset(task_dataset, list(range(100))) for task_dataset in setting.val_datasets
    ]
    setting.test_datasets = [
        Subset(task_dataset, list(range(100))) for task_dataset in setting.test_datasets
    ]
    return setting


@pytest.fixture(scope="session")
def sl_track_setting(config: Config):
    setting = SettingProxy(
        ClassIncrementalSetting,
        "sl_track",
        # dataset="synbols",
        # nb_tasks=12,
        # class_order=class_order,
        # monitor_training_performance=True,
    )
    setting.config = config
    setting.data_dir = config.data_dir
    return setting
