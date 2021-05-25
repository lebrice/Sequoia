# TODO: Create a sort of reusable fixture for the Method
# TODO: Figure out how to ACTUALLY set the checkpoint dir in pytorch-lightning!
from typing import List, Type

import pytest
from sequoia.common import Config
from sequoia.conftest import parametrize, slow
from sequoia.settings import (
    ClassIncrementalSetting,
    ContinualRLSetting,
    TraditionalSLSetting,
    IncrementalRLSetting,
    RLSetting,
    Setting,
    all_settings,
)

from .random_baseline import RandomBaselineMethod

# Use 'Method' as an alias for the actual Method cusblass under test. (since at
# the moment quite a few tests share some common code.

# List of datasets that are currently supported.
supported_datasets: List[str] = [
    "mnist",
    "fashionmnist",
    "cifar10",
    "cifar100",
    "kmnist",
    "cartpole",
]


def test_is_applicable_to_all_settings():
    settings = RandomBaselineMethod.get_applicable_settings()
    assert set(settings) == set(all_settings)


@pytest.mark.skip(
    reason="Replacing this with tests on the Settings that use this random baseline."
)
@pytest.mark.xfail(reason="TODO: This test isn't very reliable.")
@pytest.mark.timeout(60)
@parametrize("setting_type", RandomBaselineMethod.get_applicable_settings())
def test_fast_dev_run(setting_type: Type[Setting], test_dataset: str, config: Config):
    dataset = test_dataset
    if dataset not in getattr(setting_type, "available_datasets", []):
        pytest.skip(
            msg=f"Skipping dataset {dataset} since it isn't available for this setting."
        )

    # Create the Setting
    kwargs = dict(dataset=dataset)
    if issubclass(setting_type, ContinualRLSetting):
        kwargs.update(max_steps=100, test_steps_per_task=100)
    if issubclass(setting_type, IncrementalRLSetting):
        kwargs.update(nb_tasks=2)
    if issubclass(setting_type, ClassIncrementalSetting):
        kwargs = dict(nb_tasks=5)
    if issubclass(setting_type, (TraditionalSLSetting, RLSetting)):
        kwargs.pop("nb_tasks", None)
    setting: Setting = setting_type(**kwargs)
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
