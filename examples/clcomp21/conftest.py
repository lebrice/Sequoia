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
def rl_track_setting(tmp_path):
    # NOTE: Here instead of loading the `rl_track.yaml`, we create instantiate it
    # directly, because we want to reduce the length of the task for testing, and it
    # isn't currently possible to both pass a preset yaml file and also pass kwargs to
    # the SettingProxy.
    rl_track_yaml_file = "rl_track.yaml"
    
    setting = SettingProxy(
        IncrementalRLSetting,
        dataset="monsterkong",
        train_task_schedule={
            0: {"level": 0},
            1: {"level": 1},
            2: {"level": 10},
            3: {"level": 11},
            4: {"level": 20},
            5: {"level": 21},
            6: {"level": 30},
            7: {"level": 31},
        },
        steps_per_task=2_000,  # Reduced length for testing
        test_steps_per_task=2_000,
        monitor_training_performance=True,
        task_labels_at_train_time=True,
    )
    assert setting.steps_per_phase == 2000
    assert sorted(setting.train_task_schedule.keys()) == list(range(0, 16_000, 2000)) 
    return setting
