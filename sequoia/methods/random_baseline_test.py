# TODO: Create a sort of reusable fixture for the Method
# TODO: Figure out how to ACTUALLY set the checkpoint dir in pytorch-lightning!
from typing import List

from sequoia.settings import all_settings

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
