"""TODO: Tests for the SettingProxy.

"""
from typing import Type

import pytest
from sequoia.common.spaces import Image
from sequoia.common.transforms import Transforms
from sequoia.methods.baseline_method import BaselineMethod
from sequoia.methods.random_baseline import RandomBaselineMethod
from sequoia.settings import (ClassIncrementalSetting,
                              DomainIncrementalSetting, Setting,
                              TaskIncrementalRLSetting, TaskIncrementalSetting,
                              all_settings)

from .setting_proxy import EnvironmentProxy, SettingProxy


@pytest.mark.parametrize("setting_type", all_settings)
def test_spaces_match(setting_type: Type[Setting]):
    setting = setting_type()
    s_proxy = SettingProxy(setting_type)
    assert s_proxy.observation_space == setting.observation_space
    assert s_proxy.action_space == setting.action_space
    assert s_proxy.reward_space == setting.reward_space


def test_transforms_get_propagated():
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


def test_random_baseline():
    method = RandomBaselineMethod()
    setting = SettingProxy(DomainIncrementalSetting)
    results = setting.apply(method)
    assert 0.45 <= results.objective <= 0.55


@pytest.mark.timeout(120)
def test_random_baseline_SL_track():
    method = RandomBaselineMethod()
    setting = SettingProxy(ClassIncrementalSetting, dataset="synbols", nb_tasks=12)
    results = setting.apply(method)
    assert 1/48 * 0.5 <= results.objective <= 1/48 * 1.5


@pytest.mark.timeout(120)
def test_baseline_SL_track():
    """ Applies the BaselineMethod on something ressembling the SL track of the
    competition.
    """
    method = BaselineMethod(max_epochs=1, no_wandb=True)
    # import numpy as np
    # class_order = np.random.permutation(48).tolist()
    # assert False, class_order
    setting = SettingProxy(
        ClassIncrementalSetting,
        dataset="synbols",
        nb_tasks=12,
        # class_order=class_order,
    )
    results = setting.apply(method)
    assert results.to_log_dict()
    
    # TODO: Add tests for having a different ordering of test tasks vs train tasks.
    
    # results: ClassIncrementalSetting.Results
    # online_perf = results.average_online_performance()
    # assert 0.30 <= online_perf.objective <= 0.50