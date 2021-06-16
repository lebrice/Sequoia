import pytest
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, TensorDataset
from avalanche.benchmarks import nc_benchmark

from sequoia.client import SettingProxy
from sequoia.common.config import Config
from sequoia.settings.sl import (
    ClassIncrementalSetting,
    ContinualSLSetting,
    TaskIncrementalSLSetting,
    DiscreteTaskAgnosticSLSetting,
)
from sequoia.settings.sl.continual.setting import random_subset, subset
from pathlib import Path
import os


# FIXME: Overwriting the 'config' fixture from before so it's 'session' scoped instead.
@pytest.fixture(scope="session")
def config(tmp_path_factory: Path):
    test_log_dir = tmp_path_factory.mktemp("test_log_dir")
    # TODO: Set the results dir somehow with the value of this `tmp_path` fixture.
    data_dir = Path(os.environ.get("SLURM_TMPDIR", os.environ.get("DATA_DIR", "data")))
    return Config(debug=True, data_dir=data_dir, seed=123, log_dir=test_log_dir)



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
    setting = TaskIncrementalSLSetting(
        dataset="mnist", nb_tasks=5, monitor_training_performance=True,
    )
    setting.config = config
    setting.prepare_data()

    setting.setup()
    # Testing this out: Shortening the train datasets:
    setting.train_datasets = [
        random_subset(task_dataset, 100) for task_dataset in setting.train_datasets
    ]
    setting.val_datasets = [
        random_subset(task_dataset, 100) for task_dataset in setting.val_datasets
    ]
    setting.test_datasets = [
        random_subset(task_dataset, 100) for task_dataset in setting.test_datasets
    ]
    assert len(setting.train_datasets) == 5
    assert len(setting.val_datasets) == 5
    assert len(setting.test_datasets) == 5
    assert all(len(dataset) == 100 for dataset in setting.train_datasets)
    assert all(len(dataset) == 100 for dataset in setting.val_datasets)
    assert all(len(dataset) == 100 for dataset in setting.test_datasets)

    # Assert that calling setup doesn't overwrite the datasets.
    setting.setup()
    assert len(setting.train_datasets) == 5
    assert len(setting.val_datasets) == 5
    assert len(setting.test_datasets) == 5
    assert all(len(dataset) == 100 for dataset in setting.train_datasets)
    assert all(len(dataset) == 100 for dataset in setting.val_datasets)
    assert all(len(dataset) == 100 for dataset in setting.test_datasets)

    return setting


@pytest.fixture(scope="session")
def short_class_incremental_setting(config: Config):
    setting = ClassIncrementalSetting(
        dataset="mnist", nb_tasks=5, monitor_training_performance=True,
    )
    setting.config = config
    setting.prepare_data()
    setting.setup()

    # Testing this out: Shortening the train datasets:
    setting.train_datasets = [
        random_subset(task_dataset, 100) for task_dataset in setting.train_datasets
    ]
    setting.val_datasets = [
        random_subset(task_dataset, 100) for task_dataset in setting.val_datasets
    ]
    setting.test_datasets = [
        random_subset(task_dataset, 100) for task_dataset in setting.test_datasets
    ]
    assert len(setting.train_datasets) == 5
    assert len(setting.val_datasets) == 5
    assert len(setting.test_datasets) == 5
    assert all(len(dataset) == 100 for dataset in setting.train_datasets)
    assert all(len(dataset) == 100 for dataset in setting.val_datasets)
    assert all(len(dataset) == 100 for dataset in setting.test_datasets)

    # Assert that calling setup doesn't overwrite the datasets.
    setting.setup()
    assert len(setting.train_datasets) == 5
    assert len(setting.val_datasets) == 5
    assert len(setting.test_datasets) == 5
    assert all(len(dataset) == 100 for dataset in setting.train_datasets)
    assert all(len(dataset) == 100 for dataset in setting.val_datasets)
    assert all(len(dataset) == 100 for dataset in setting.test_datasets)
    return setting


@pytest.fixture(scope="session")
def short_continual_sl_setting(config: Config):
    setting = ContinualSLSetting(dataset="mnist", monitor_training_performance=True,)
    setting.config = config
    setting.prepare_data()
    setting.setup()

    # Testing this out: Shortening the train datasets:
    setting.train_datasets = [
        random_subset(task_dataset, 100) for task_dataset in setting.train_datasets
    ]
    setting.val_datasets = [
        random_subset(task_dataset, 100) for task_dataset in setting.val_datasets
    ]
    setting.test_datasets = [
        random_subset(task_dataset, 100) for task_dataset in setting.test_datasets
    ]
    assert len(setting.train_datasets) == 5
    assert len(setting.val_datasets) == 5
    assert len(setting.test_datasets) == 5
    assert all(len(dataset) == 100 for dataset in setting.train_datasets)
    assert all(len(dataset) == 100 for dataset in setting.val_datasets)
    assert all(len(dataset) == 100 for dataset in setting.test_datasets)

    # Assert that calling setup doesn't overwrite the datasets.
    setting.setup()
    assert len(setting.train_datasets) == 5
    assert len(setting.val_datasets) == 5
    assert len(setting.test_datasets) == 5
    assert all(len(dataset) == 100 for dataset in setting.train_datasets)
    assert all(len(dataset) == 100 for dataset in setting.val_datasets)
    assert all(len(dataset) == 100 for dataset in setting.test_datasets)
    return setting


@pytest.fixture(scope="session")
def short_discrete_task_agnostic_sl_setting(config: Config):
    setting = DiscreteTaskAgnosticSLSetting(
        dataset="mnist", monitor_training_performance=True,
    )
    setting.config = config
    setting.prepare_data()
    setting.setup()

    # Testing this out: Shortening the train datasets:
    setting.train_datasets = [
        random_subset(task_dataset, 100) for task_dataset in setting.train_datasets
    ]
    setting.val_datasets = [
        random_subset(task_dataset, 100) for task_dataset in setting.val_datasets
    ]
    setting.test_datasets = [
        random_subset(task_dataset, 100) for task_dataset in setting.test_datasets
    ]
    assert len(setting.train_datasets) == 5
    assert len(setting.val_datasets) == 5
    assert len(setting.test_datasets) == 5
    assert all(len(dataset) == 100 for dataset in setting.train_datasets)
    assert all(len(dataset) == 100 for dataset in setting.val_datasets)
    assert all(len(dataset) == 100 for dataset in setting.test_datasets)

    # Assert that calling setup doesn't overwrite the datasets.
    setting.setup()
    assert len(setting.train_datasets) == 5
    assert len(setting.val_datasets) == 5
    assert len(setting.test_datasets) == 5
    assert all(len(dataset) == 100 for dataset in setting.train_datasets)
    assert all(len(dataset) == 100 for dataset in setting.val_datasets)
    assert all(len(dataset) == 100 for dataset in setting.test_datasets)
    return setting


@pytest.fixture(scope="session")
def short_sl_track_setting(config: Config):
    setting = SettingProxy(
        ClassIncrementalSetting,
        "sl_track",
        # dataset="synbols",
        # nb_tasks=12,
        # class_order=class_order,
        # monitor_training_performance=True,
    )
    setting.config = config
    # TODO: This could be a bit more convenient.
    setting.data_dir = config.data_dir
    assert setting.config == config
    assert setting.data_dir == config.data_dir
    assert setting.nb_tasks == 12

    # For now we'll just shorten the tests by shortening the datasets.
    samples_per_task = 100
    setting.batch_size = 10

    setting.setup()
    # Testing this out: Shortening the train datasets:
    setting.train_datasets = [
        random_subset(task_dataset, samples_per_task)
        for task_dataset in setting.train_datasets
    ]
    setting.val_datasets = [
        random_subset(task_dataset, samples_per_task)
        for task_dataset in setting.val_datasets
    ]
    setting.test_datasets = [
        random_subset(task_dataset, samples_per_task)
        for task_dataset in setting.test_datasets
    ]
    assert len(setting.train_datasets) == setting.nb_tasks
    assert len(setting.val_datasets) == setting.nb_tasks
    assert len(setting.test_datasets) == setting.nb_tasks
    assert all(len(dataset) == samples_per_task for dataset in setting.train_datasets)
    assert all(len(dataset) == samples_per_task for dataset in setting.val_datasets)
    assert all(len(dataset) == samples_per_task for dataset in setting.test_datasets)

    # Assert that calling setup doesn't overwrite the datasets.
    setting.setup()

    assert len(setting.train_datasets) == setting.nb_tasks
    assert len(setting.val_datasets) == setting.nb_tasks
    assert len(setting.test_datasets) == setting.nb_tasks
    assert all(len(dataset) == samples_per_task for dataset in setting.train_datasets)
    assert all(len(dataset) == samples_per_task for dataset in setting.val_datasets)
    assert all(len(dataset) == samples_per_task for dataset in setting.test_datasets)

    return setting
