import pytest
from .multihead_classifier import ExampleTaskInferenceMethod
from sequoia.client.setting_proxy import SettingProxy
from sequoia.settings.passive import ClassIncrementalSetting
from sequoia.settings.active import IncrementalRLSetting


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
