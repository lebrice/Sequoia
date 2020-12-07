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

# List of datasets that are currently supported.
supported_datasets: List[str] = [
    "mnist", "fashion_mnist", "cifar10", "cifar100", "kmnist", "cartpole"
]


def test_is_applicable_to_all_settings():
    settings = RandomBaselineMethod.get_applicable_settings()
    assert set(settings) == set(all_settings)


@pytest.mark.timeout(30)
@parametrize("setting_type", RandomBaselineMethod.get_applicable_settings())
def test_fast_dev_run(setting_type: Type[Setting],
                      test_dataset: str,
                      config: Config):
    dataset = test_dataset
    if dataset not in getattr(setting_type, "available_datasets", []):
        pytest.skip(msg=f"dataset {dataset} isn't available for this setting.")
    
    # Create the Setting
    setting: Setting = setting_type(dataset=dataset)
    
    # TODO: IDEA: Let the settings configure themselves for quick debugging.
    if isinstance(setting, ContinualRLSetting):
        setting.max_steps = 100

    # Create the Method
    method = RandomBaselineMethod()
    
    results = setting.apply(method, config=config)
    method.validate_results(setting, results)
    