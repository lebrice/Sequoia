from functools import singledispatch
from pathlib import Path
# TODO: Create a sort of reusable fixture for the Method
# TODO: Figure out how to ACTUALLY set the checkpoint dir in pytorch-lightning!
from typing import Any, Callable, List, Type

import pytest

from common import ClassificationMetrics, Config
from conftest import get_dataset_params, parametrize, slow
from settings import (ClassIncrementalResults, ClassIncrementalRLSetting,
                      ClassIncrementalSetting, ContinualRLSetting, IIDSetting,
                      Results, RLSetting, Setting, TaskIncrementalResults,
                      TaskIncrementalRLSetting, TaskIncrementalSetting,
                      all_settings)

from .random_baseline import RandomBaselineMethod

# Use 'Method' as an alias for the actual Method cusblass under test. (since at
# the moment quite a few tests share some common code.
Method = RandomBaselineMethod

# List of datasets that are currently supported.
supported_datasets: List[str] = [
    "mnist", "fashion_mnist", "cifar10", "cifar100", "kmnist", "cartpole"
]


def test_get_applicable_settings():
    settings = Method.get_applicable_settings()
    assert set(settings) == set(all_settings)

# Reuse the method accross all tests below
@pytest.fixture(scope="module") 
def method(tmp_path_factory: Callable[[str], Path]):
    log_dir = tmp_path_factory.mktemp("log_dir")
    return RandomBaselineMethod.from_args(f"""
        --debug
        --log_dir_root {log_dir}
        --default_root_dir {log_dir}
        --knn_samples 0
        --seed 123
        --fast_dev_run
        --batch_size 16
        """
        # TODO: There is something weird going on here. We don't get chance
        # levels on IID cifar100 or even Mnist when in the IID setting when
        # limiting the number of batches to some number..
        # --limit_train_batches 1
        # --limit_val_batches 10
        # --limit_test_batches 10
    )

# @parametrize("dataset", get_dataset_params(Method, supported_datasets))
@slow
@parametrize("setting_type", Method.get_applicable_settings())
def test_fast_dev_run(method: RandomBaselineMethod, setting_type: Type[Setting], test_dataset: str, config: Config):
    dataset = test_dataset
    if dataset not in getattr(setting_type, "available_datasets", []):
        pytest.skip(msg=f"dataset {dataset} isn't available for this setting.")
    # Instantiate the setting
    setting: Setting = setting_type(dataset=dataset, nb_tasks=5)
    if isinstance(setting, IIDSetting):
        assert setting.nb_tasks == 1
    results: Results = setting.apply(method, config)
    method.validate_results(setting, results)


def test_fast_dev_run_multihead(tmp_path: Path, config: Config):
    setting = TaskIncrementalSetting(
        dataset="mnist",
        increment=2,
    )
    method: RandomBaselineMethod = RandomBaselineMethod.from_args(f"""
        --debug
        --fast_dev_run
        --default_root_dir {tmp_path}
        --multihead True
        --batch_size 10
    """)
    results: TaskIncrementalResults = setting.apply(method, config=config)
    metrics = results.average_metrics_per_task
    assert metrics
    for metric in metrics:
        if isinstance(metric, ClassificationMetrics):
            assert metric.confusion_matrix.shape == (2, 2)
    method.validate_results(setting, results)
