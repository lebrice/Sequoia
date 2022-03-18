"""TODO: Tests for the SettingProxy.

"""
from functools import partial
from typing import ClassVar, Type

import numpy as np
import pytest
from gym import spaces

from sequoia.common.metrics.rl_metrics import EpisodeMetrics
from sequoia.common.spaces import Image, Sparse
from sequoia.common.transforms import Transforms
from sequoia.conftest import slow
from sequoia.methods.base_method import BaseMethod
from sequoia.methods.method_test import key_fn
from sequoia.methods.random_baseline import RandomBaselineMethod
from sequoia.settings import Setting, all_settings
from sequoia.settings.rl import IncrementalRLSetting, TaskIncrementalRLSetting
from sequoia.settings.rl.continual.setting import ContinualRLSetting
from sequoia.settings.rl.continual.setting_test import (
    TestContinualRLSetting as ContinualRLSettingTests,
)
from sequoia.settings.sl import ClassIncrementalSetting, DomainIncrementalSLSetting
from sequoia.settings.sl.continual.setting import ContinualSLSetting
from sequoia.settings.sl.continual.setting_test import (
    TestContinualSLSetting as ContinualSLSettingTests,
)

from .setting_proxy import SettingProxy


@pytest.mark.parametrize("setting_type", sorted(all_settings, key=key_fn))
def test_spaces_match(setting_type: Type[Setting]):
    setting = setting_type()
    s_proxy = SettingProxy(setting_type)
    assert s_proxy.observation_space == setting.observation_space
    assert s_proxy.action_space == setting.action_space
    assert s_proxy.reward_space == setting.reward_space


def test_transforms_get_propagated():
    for setting in [
        TaskIncrementalRLSetting(dataset="MetaMonsterKong-v0"),
        SettingProxy(TaskIncrementalRLSetting, dataset="MetaMonsterKong-v0"),
    ]:
        assert setting.observation_space.x == Image(0, 255, shape=(64, 64, 3), dtype=np.uint8)
        setting.transforms.append(Transforms.to_tensor)
        setting.transforms.append(Transforms.resize_32x32)
        # TODO: The observation space doesn't update directly in RL whenever the
        # transforms are changed.
        assert setting.observation_space.x == Image(0, 1, shape=(3, 32, 32))
        assert setting.train_dataloader().reset().x.shape == (3, 32, 32)


class TestContinualSLSettingProxy(ContinualSLSettingTests):
    Setting: ClassVar[Type[Setting]] = partial(SettingProxy, ContinualSLSetting)


class TestContinualRLSettingProxy(ContinualRLSettingTests):
    Setting: ClassVar[Type[Setting]] = partial(SettingProxy, ContinualRLSetting)


@pytest.mark.timeout(30)
def test_random_baseline(config):
    method = RandomBaselineMethod()
    setting = SettingProxy(DomainIncrementalSLSetting, config=config)
    results = setting.apply(method, config=config)
    # domain incremental mnist: 2 classes per task -> chance accuracy of 50%.
    assert 0.45 <= results.objective <= 0.55


@pytest.mark.timeout(180)
def test_random_baseline_rl():
    method = RandomBaselineMethod()
    setting = SettingProxy(
        IncrementalRLSetting,
        dataset="monsterkong",
        monitor_training_performance=True,
        # observe_state_directly=False, ## TODO: Make sure this doesn't change anything.
        train_steps_per_task=1_000,
        test_steps_per_task=1_000,
        train_task_schedule={
            0: {"level": 0},
            1: {"level": 1},
            2: {"level": 10},
            3: {"level": 11},
            4: {"level": 0},
        },
        # Interesting problem: Will it always do at least an entire episode here per
        # env?
        # batch_size=2,
        # num_workers=0,
    )
    assert setting.train_max_steps == 4_000
    assert setting.test_max_steps == 4_000
    results: IncrementalRLSetting.Results[EpisodeMetrics] = setting.apply(method)
    assert 20 <= results.average_final_performance.mean_reward_per_episode


@pytest.mark.timeout(120)
def test_random_baseline_SL_track():
    method = RandomBaselineMethod()
    setting = SettingProxy(ClassIncrementalSetting, dataset="synbols", nb_tasks=12)
    results = setting.apply(method)
    assert 1 / 48 * 0.5 <= results.objective <= 1 / 48 * 1.5


@slow
@pytest.mark.timeout(300)
def test_baseline_SL_track(config):
    """Applies the BaseMethod on something ressembling the SL track of the
    competition.
    """
    method = BaseMethod(max_epochs=1)
    import numpy as np

    class_order = np.random.permutation(48).tolist()
    setting = SettingProxy(
        ClassIncrementalSetting,
        dataset="synbols",
        nb_tasks=12,
        class_order=class_order,
    )
    results = setting.apply(method, config)
    assert results.to_log_dict()

    # TODO: Add tests for having a different ordering of test tasks vs train tasks.
    results: ClassIncrementalSetting.Results
    online_perf = results.average_online_performance
    assert 0.30 <= online_perf.objective <= 0.65
    final_perf = results.average_final_performance
    assert 0.02 <= final_perf.objective <= 0.06


def test_rl_track_setting_is_correct():
    setting = SettingProxy(
        IncrementalRLSetting,
        "rl_track",
    )
    assert setting.nb_tasks == 8
    assert setting.dataset == "MetaMonsterKong-v0"
    assert setting.observation_space == spaces.Dict(
        x=Image(0, 1, (3, 64, 64), dtype=np.float32),
        task_labels=Sparse(spaces.Discrete(8)),
    )
    assert setting.action_space == spaces.Discrete(6)
    # TODO: The reward range of the MetaMonsterKongEnv is (0, 50), which seems wrong.
    # This isn't really a big deal though.
    # assert setting.reward_space == spaces.Box(0, 100, shape=(), dtype=np.float32)
    assert setting.steps_per_task == 200_000
    assert setting.test_steps_per_task == 10_000
    assert setting.known_task_boundaries_at_train_time is True
    assert setting.known_task_boundaries_at_test_time is False
    assert setting.monitor_training_performance is True
    assert setting.train_transforms == [Transforms.to_tensor, Transforms.three_channels]
    assert setting.val_transforms == [Transforms.to_tensor, Transforms.three_channels]
    assert setting.test_transforms == [Transforms.to_tensor, Transforms.three_channels]

    train_env = setting.train_dataloader()
    assert train_env.observation_space == spaces.Dict(
        x=Image(0, 1, (3, 64, 64), dtype=np.float32),
        task_labels=spaces.Discrete(8),
    )
    assert train_env.reset() in train_env.observation_space

    valid_env = setting.val_dataloader()
    assert valid_env.observation_space == spaces.Dict(
        x=Image(0, 1, (3, 64, 64), dtype=np.float32),
        task_labels=spaces.Discrete(8),
    )

    # IDEA: Prevent submissions from calling the test_dataloader method or accessing the
    # test_env / test_dataset property?
    with pytest.raises(RuntimeError):
        test_env = setting.test_dataloader()
        test_env.reset()

    with pytest.raises(RuntimeError):
        test_env = setting.test_env
        test_env.reset()


def test_sl_track_setting_is_correct():
    setting = SettingProxy(
        ClassIncrementalSetting,
        "sl_track",
    )
    assert setting.nb_tasks == 12
    assert setting.dataset == "synbols"
    assert setting.observation_space == spaces.Dict(
        x=Image(0, 1, (3, 32, 32), dtype=np.float32),
        task_labels=spaces.Discrete(12),
    )
    assert setting.n_classes_per_task == 4
    assert setting.action_space == spaces.Discrete(48)
    assert setting.reward_space == spaces.Discrete(48)
    assert setting.known_task_boundaries_at_train_time is True
    assert setting.known_task_boundaries_at_test_time is False
    assert setting.monitor_training_performance is True
    assert setting.train_transforms == [Transforms.to_tensor, Transforms.three_channels]
    assert setting.val_transforms == [Transforms.to_tensor, Transforms.three_channels]
    assert setting.test_transforms == [Transforms.to_tensor, Transforms.three_channels]
