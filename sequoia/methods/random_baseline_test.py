from functools import singledispatch
from pathlib import Path
# TODO: Create a sort of reusable fixture for the Method
# TODO: Figure out how to ACTUALLY set the checkpoint dir in pytorch-lightning!
from typing import Any, Callable, List, Type

import pytest

from sequoia.common import ClassificationMetrics, Config
from sequoia.conftest import get_dataset_params, parametrize, slow
from sequoia.settings.assumptions.incremental import IncrementalSetting
from sequoia.settings import (ClassIncrementalResults, IncrementalRLSetting,
                      ClassIncrementalSetting, ContinualRLSetting, IIDSetting,
                      Results, RLSetting, Setting, TaskIncrementalResults,
                      TaskIncrementalRLSetting, TaskIncrementalSetting,
                      all_settings)

from .random_baseline import RandomBaselineMethod

# Use 'Method' as an alias for the actual Method cusblass under test. (since at
# the moment quite a few tests share some common code.

# List of datasets that are currently supported.
supported_datasets: List[str] = [
    "mnist", "fashion_mnist", "cifar10", "cifar100", "kmnist", "cartpole"
]


def test_is_applicable_to_all_settings():
    settings = RandomBaselineMethod.get_applicable_settings()
    assert set(settings) == set(all_settings)

from sequoia.conftest import slow

# This is a very slow test, because it actually iterates through the entire test set for each task.
@pytest.mark.timeout(60)
@parametrize("setting_type", RandomBaselineMethod.get_applicable_settings())
def test_fast_dev_run(setting_type: Type[Setting],
                      test_dataset: str,
                      config: Config):
    dataset = test_dataset
    if dataset not in getattr(setting_type, "available_datasets", []):
        pytest.skip(msg=f"dataset {dataset} isn't available for this setting.")

    # Create the Setting
    setting: Setting = setting_type(dataset=dataset)
    # TODO: Do we need to pass anything else here to 'shorten' the run?
    # Create the Method
    method = RandomBaselineMethod()
    if isinstance(setting, ClassIncrementalSetting):
        method.batch_size = 64
    elif isinstance(setting, ContinualRLSetting):
        method.batch_size = None
        setting.max_steps = 100
    
    results = setting.apply(method, config=config)
    method.validate_results(setting, results)
    