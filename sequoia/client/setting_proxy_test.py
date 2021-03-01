"""TODO: Tests for the SettingProxy.

"""
from typing import Type

import pytest
from sequoia.settings import (ClassIncrementalSetting, Setting,
                              TaskIncrementalRLSetting, TaskIncrementalSetting, all_settings)

from .setting_proxy import EnvironmentProxy, SettingProxy
from sequoia.common.spaces import Image
from sequoia.common.transforms import Transforms


@pytest.mark.parametrize("setting_type", all_settings)
def test_spaces_match(setting_type: Type[Setting]):
    setting = setting_type()
    s_proxy = SettingProxy(setting_type)
    assert s_proxy.observation_space == setting.observation_space
    assert s_proxy.action_space == setting.action_space
    assert s_proxy.reward_space == setting.reward_space


def test_transforms_get_propagated():
    setting = TaskIncrementalRLSetting(dataset="cartpole")
    for setting in [
        TaskIncrementalRLSetting(dataset="cartpole"), 
        SettingProxy(TaskIncrementalRLSetting, dataset="cartpole"),
    ]:
        assert setting.observation_space.x == Image(0, 1, shape=(3, 400, 600))
        setting.train_transforms.append(Transforms.resize_64x64)
        # TODO: The observation space doesn't update directly in RL whenever the
        # transforms are changed.
        # assert setting.observation_space.x == Image(0, 1, shape=(3, 64, 64))
        assert setting.train_dataloader().reset().x.shape == (3, 64, 64)
