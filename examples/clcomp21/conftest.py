import pytest
import sys

from sequoia.client.setting_proxy import SettingProxy
from sequoia.settings.passive import ClassIncrementalSetting
from sequoia.settings.active import IncrementalRLSetting, RLSetting


@pytest.fixture()
def mnist_setting():
    return SettingProxy(
        ClassIncrementalSetting,
        dataset="mnist",
        monitor_training_performance=True,
    )


@pytest.fixture()
def fashion_mnist_setting():
    return SettingProxy(
        ClassIncrementalSetting,
        dataset="fashionmnist",
        monitor_training_performance=True,
    )


@pytest.fixture()
def sl_track_setting():
    setting = SettingProxy(
        ClassIncrementalSetting,
        "sl_track",
        # dataset="synbols",
        # nb_tasks=12,
        # class_order=class_order,
        # monitor_training_performance=True,
    )
    return setting


@pytest.fixture()
def cartpole_state_setting():
    setting = SettingProxy(
        RLSetting,
        dataset="cartpole",
        observe_state_directly=True,
        max_steps=5_000,
        test_steps=2_000,
        monitor_training_performance=True,
    )
    return setting


@pytest.fixture()
def incremental_cartpole_state_setting():
    setting = SettingProxy(
        IncrementalRLSetting,
        dataset="cartpole",
        observe_state_directly=True,
        max_steps=10_000,
        nb_tasks=2,
        test_steps=2_000,
        monitor_training_performance=True,
    )
    return setting


@pytest.fixture()
def rl_track_setting():
    setting = SettingProxy(
        IncrementalRLSetting,
        "rl_track",
        steps_per_task=2_000,  # just for testing.
        test_steps_per_task=2_000,  # just for testing.
        monitor_training_performance=True,
    )
    return setting
