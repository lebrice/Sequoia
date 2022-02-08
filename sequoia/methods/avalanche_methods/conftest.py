from pathlib import Path

import pytest
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

from sequoia.common.config import Config

collect_ignore = []
collect_ignore_glob = []
try:
    from avalanche.training.strategies import BaseStrategy  # type: ignore
except ImportError:
    # pytest.skip(reason="Needs avalanche", allow_module_level=True)
    collect_ignore_glob.append("sequoia/methods/avalanche/**.py")


# FIXME: Overwriting the 'config' fixture from before so it's 'session' scoped instead.
@pytest.fixture(scope="session")
def config(tmp_path_factory):
    test_log_dir = tmp_path_factory.mktemp("test_log_dir")
    return Config(debug=True, seed=123, log_dir=test_log_dir)


@pytest.fixture(scope="session")
def fast_scenario(use_task_labels=False, shuffle=True):
    """Copied directly from Avalanche in "tests/unit_tests_utils.py".

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
    from avalanche.benchmarks import nc_benchmark  # type: ignore  
    train_dataset = TensorDataset(train_X, train_y)
    test_dataset = TensorDataset(test_X, test_y)
    my_nc_benchmark = nc_benchmark(
        train_dataset, test_dataset, 5, task_labels=use_task_labels, shuffle=shuffle
    )
    return my_nc_benchmark
