import pytest
import torch
from sklearn.datasets import make_classification
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
def session_config(tmp_path_factory: Path):
    test_log_dir = tmp_path_factory.mktemp("test_log_dir")
    # TODO: Set the results dir somehow with the value of this `tmp_path` fixture.
    data_dir = Path(os.environ.get("SLURM_TMPDIR", os.environ.get("DATA_DIR", "data")))
    return Config(debug=True, data_dir=data_dir, seed=123, log_dir=test_log_dir)



@pytest.fixture(scope="session")
def short_class_incremental_setting(session_config: Config):
    setting = ClassIncrementalSetting(
        dataset="mnist", nb_tasks=5, monitor_training_performance=True,
    )
    setting.config = session_config
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
def short_continual_sl_setting(session_config: Config):
    setting = ContinualSLSetting(dataset="mnist", monitor_training_performance=True,)
    setting.config = session_config
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
def short_discrete_task_agnostic_sl_setting(session_config: Config):
    setting = DiscreteTaskAgnosticSLSetting(
        dataset="mnist", monitor_training_performance=True,
    )
    setting.config = session_config
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
def short_task_incremental_setting(session_config: Config):
    setting = TaskIncrementalSLSetting(
        dataset="mnist", nb_tasks=5, monitor_training_performance=True,
    )
    setting.config = session_config
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
def short_sl_track_setting(session_config: Config):
    setting = SettingProxy(
        ClassIncrementalSetting,
        "sl_track",
        # dataset="synbols",
        # nb_tasks=12,
        # class_order=class_order,
        # monitor_training_performance=True,
    )
    setting.config = session_config
    # TODO: This could be a bit more convenient.
    setting.data_dir = session_config.data_dir
    assert setting.config == session_config
    assert setting.data_dir == session_config.data_dir
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
