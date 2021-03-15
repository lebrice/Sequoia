import pytest
import sys

from sequoia.client.setting_proxy import SettingProxy
from sequoia.settings.passive import ClassIncrementalSetting
from sequoia.settings.active import IncrementalRLSetting


@pytest.fixture()
def mnist_setting():
    return SettingProxy(
        ClassIncrementalSetting,
        dataset="mnist",
    )


@pytest.fixture()
def fashion_mnist_setting():
    return SettingProxy(
        ClassIncrementalSetting,
        dataset="fashionmnist",
    )


@pytest.fixture()
def sl_track_setting():
    setting = SettingProxy(
        ClassIncrementalSetting,
        "sl_track",
        # dataset="synbols",
        # nb_tasks=12,
        # class_order=class_order,
    )
    return setting


@pytest.fixture()
def rl_track_setting():
    setting = SettingProxy(
        IncrementalRLSetting,
        # "rl_track", # TODO: Levels 0-20 work for now in MonsterKong.
        "rl_track",
        steps_per_task=2_000,  # just for testing.
        test_steps_per_task=2_000,  # just for testing.
        # dataset="synbols",
        # nb_tasks=12,
        # class_order=class_order,
    )
    return setting


def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true", default=False)


slow = pytest.mark.skipif(
    "--slow" not in sys.argv,
    reason="This test is slow so we only run it when necessary."
)
