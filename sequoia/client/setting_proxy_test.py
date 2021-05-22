"""TODO: Tests for the SettingProxy.

"""
from typing import Type

import numpy as np
import pytest
from gym import spaces
from sequoia.common.spaces import Image, NamedTupleSpace, Sparse
from sequoia.common.transforms import Transforms
from sequoia.methods.baseline_method import BaselineMethod
from sequoia.methods.random_baseline import RandomBaselineMethod
from sequoia.settings import Setting, all_settings
from sequoia.settings.active import IncrementalRLSetting, TaskIncrementalRLSetting
from sequoia.settings.passive import ClassIncrementalSetting, DomainIncrementalSetting
from sequoia.conftest import slow
from sequoia.common.metrics.rl_metrics import EpisodeMetrics

from .setting_proxy import SettingProxy


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


@pytest.mark.timeout(60)
def test_random_baseline():
    method = RandomBaselineMethod()
    setting = SettingProxy(DomainIncrementalSetting)
    results = setting.apply(method)
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
        steps_per_task=1_000,
        test_steps_per_task=1_000,
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
        # Interesting problem: Will it always do at least an entire episode here per
        # env?
        # batch_size=2,
        # num_workers=0,
    )
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
    """ Applies the BaselineMethod on something ressembling the SL track of the
    competition.
    """
    method = BaselineMethod(max_epochs=1)
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
    setting = SettingProxy(IncrementalRLSetting, "rl_track",)
    assert setting.nb_tasks == 8
    assert setting.dataset == "MetaMonsterKong-v0"
    assert setting.observation_space == NamedTupleSpace(
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
    assert train_env.observation_space == NamedTupleSpace(
        x=Image(0, 1, (3, 64, 64), dtype=np.float32), task_labels=spaces.Discrete(8),
    )
    assert train_env.reset() in train_env.observation_space

    valid_env = setting.val_dataloader()
    assert valid_env.observation_space == NamedTupleSpace(
        x=Image(0, 1, (3, 64, 64), dtype=np.float32), task_labels=spaces.Discrete(8),
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
    setting = SettingProxy(ClassIncrementalSetting, "sl_track",)
    assert setting.nb_tasks == 12
    assert setting.dataset == "synbols"
    assert setting.observation_space == NamedTupleSpace(
        x=Image(0, 1, (3, 32, 32), dtype=np.float32), task_labels=spaces.Discrete(12),
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
