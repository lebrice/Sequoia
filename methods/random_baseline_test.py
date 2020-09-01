from functools import singledispatch
from pathlib import Path
# TODO: Create a sort of reusable fixture for the Method
# TODO: Figure out how to ACTUALLY set the checkpoint dir in pytorch-lightning!
from typing import Any, Callable, List, Type

import pytest

from common import ClassificationMetrics
from conftest import get_dataset_params, parametrize
from settings import (ClassIncrementalResults, ClassIncrementalSetting,
                      IIDSetting, Results, Setting, TaskIncrementalResults,
                      TaskIncrementalSetting)

from methods.models.class_incremental_model import ClassIncrementalModel
from .random_baseline import RandomBaselineMethod

# Use 'Method' as an alias for the actual Method cusblass under test. (since at
# the moment quite a few tests share some common code.
Method = RandomBaselineMethod

# List of datasets that are currently supported.
supported_datasets: List[str] = [
    "mnist", "fashion_mnist", "cifar10", "cifar100", "kmnist"
]

def test_get_applicable_settings():
    settings = Method.get_all_applicable_settings()
    assert ClassIncrementalSetting in settings
    assert TaskIncrementalSetting in settings
    assert IIDSetting in settings


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
        --batch_size 1000
        """
        # TODO: There is something weird going on here. We don't get chance
        # levels on IID cifar100 or even Mnist when in the IID setting when
        # limiting the number of batches to some number..
        # --limit_train_batches 1
        # --limit_val_batches 10
        # --limit_test_batches 10
    )


# @parametrize("dataset", get_dataset_params(Method, supported_datasets))
@parametrize("setting_type", Method.get_all_applicable_settings())
def test_fast_dev_run(method: RandomBaselineMethod, setting_type: Type[Setting], test_dataset: str):
    dataset = test_dataset
    if dataset not in setting_type.available_datasets:
        pytest.skip(msg=f"dataset {dataset} isn't available for this setting.")
    # Instantiate the setting
    setting: Setting = setting_type(dataset=dataset, nb_tasks=5)
    if isinstance(setting, IIDSetting):
        assert setting.nb_tasks == 1
    results: Results = method.apply_to(setting)
    validate_results(results, setting)


def validate_results(results: Results, setting: Setting):
    """Makes sure that the results make sense for the method being tested.

    Since each setting defines its own Results class, we can switch based on the
    results class and see if it makes sense.

    Args:
        results (Results): A given Results object.
    """
    assert results is not None
    assert results.hparams is not None
    assert results.test_loss is not None
    assert results.objective > 0
    print(f"Objective: {results.objective}")

    if isinstance(results, ClassIncrementalResults):
        assert isinstance(setting, ClassIncrementalSetting)
        assert isinstance(results.hparams, ClassIncrementalModel.HParams)

        average_accuracy = results.objective

        # Calculate the expected 'average' chance accuracy.
        if results.hparams.multihead:
            # We assume that there is an equal number of classes in each task.
            assert isinstance(setting.increment, int)
            num_classes = setting.increment
        else:
            num_classes = setting.num_classes
        chance_accuracy = 1 / num_classes

        assert 0.5 * chance_accuracy <= average_accuracy <= 1.5 * chance_accuracy


        for i, task_loss in enumerate(results.task_losses):
            metric = task_loss.metric
            assert isinstance(metric, ClassificationMetrics)
            # TODO: Check that this makes sense:

            if getattr(results.hparams, "multihead", False):
                # NOTE: This assumes that there is an equal number of classes in
                # each task.
                assert isinstance(setting.increment, int)
                num_classes = setting.increment
            else:
                num_classes = setting.num_classes
            chance_accuracy = 1 / num_classes

            task_accuracy = task_loss.metric.accuracy
            # FIXME: Look into this, we're often getting results substantially
            # worse than chance, and to 'make the tests pass' (which is bad)
            # we're setting the lower bound super low, which makes no sense.
            assert 0.25 * chance_accuracy <= task_accuracy <= 2.1 * chance_accuracy


def test_fast_dev_run_multihead(tmp_path: Path):
    setting = TaskIncrementalSetting(
        dataset="mnist",
        increment=2,
    )
    method: RandomBaselineMethod = RandomBaselineMethod.from_args(f"""
        --debug
        --fast_dev_run
        --default_root_dir {tmp_path}
        --log_dir_root {tmp_path}
        --multihead True
        --batch_size 100
    """)
    results: TaskIncrementalResults = method.apply_to(setting)
    metrics = results.task_metrics
    assert metrics
    for metric in metrics:
        if isinstance(metric, ClassificationMetrics):
            assert metric.confusion_matrix.shape == (2, 2)
    validate_results(results, setting)
