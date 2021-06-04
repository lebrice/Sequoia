from typing import ClassVar, Type

import pytest
from sequoia.settings import Setting
from sequoia.settings.assumptions.incremental_test import DummyMethod as _DummyMethod
from sequoia.settings.rl.envs import (
    ATARI_PY_INSTALLED,
    MONSTERKONG_INSTALLED,
    MUJOCO_INSTALLED,
)

from ..continual.setting_test import TestContinualRLSetting as ContinualRLSettingTests
from ..continual.setting_test import make_dataset_fixture
from .setting import DiscreteTaskAgnosticRLSetting


class TestDiscreteTaskAgnosticRLSetting(ContinualRLSettingTests):
    Setting: ClassVar[Type[Setting]] = DiscreteTaskAgnosticRLSetting
    dataset: pytest.fixture = make_dataset_fixture(DiscreteTaskAgnosticRLSetting)

    # @pytest.fixture(
    #     params=list(DiscreteTaskAgnosticRLSetting.available_datasets.keys()),
    #     scope="session",
    # )
    # def dataset(self, request):
    #     dataset = request.param
    #     return dataset

    @pytest.fixture(params=[1, 3, 5])
    def nb_tasks(self, request):
        n = request.param
        return n

    @pytest.fixture()
    def setting_kwargs(self, dataset: str, nb_tasks: int):
        """ Fixture used to pass keyword arguments when creating a Setting. """
        return {"dataset": dataset, "nb_tasks": nb_tasks}


def test_passing_task_schedule_sets_other_attributes_correctly():
    setting = DiscreteTaskAgnosticRLSetting(
        dataset="CartPole-v0",
        train_task_schedule={
            0: {"gravity": 5.0},
            100: {"gravity": 10.0},
            200: {"gravity": 20.0},
        },
        test_max_steps=10_000,
    )
    assert setting.phases == 1
    assert setting.nb_tasks == 2
    assert setting.train_steps_per_task == 100
    assert setting.test_task_schedule == {
        0: {"gravity": 5.0},
        5_000: {"gravity": 10.0},
        10_000: {"gravity": 20.0},
    }
    assert setting.test_max_steps == 10_000
    # assert setting.test_steps_per_task == 5_000

    setting = DiscreteTaskAgnosticRLSetting(
        dataset="CartPole-v0",
        train_task_schedule={
            0: {"gravity": 5.0},
            100: {"gravity": 10.0},
            200: {"gravity": 20.0},
        },
        test_max_steps=2000,
        test_steps_per_task=100,
    )
    assert setting.phases == 1
    # assert setting.nb_tasks == 2
    # assert setting.steps_per_task == 100
    assert setting.test_task_schedule == {
        0: {"gravity": 5.0},
        1000: {"gravity": 10.0},
        2000: {"gravity": 20.0},
    }
    assert setting.test_max_steps == 2000
    # assert setting.test_steps_per_task == 100


def test_fit_and_on_task_switch_calls():
    setting = DiscreteTaskAgnosticRLSetting(
        dataset="CartPole-v0",
        # nb_tasks=5,
        # steps_per_task=100,
        train_max_steps=500,
        test_max_steps=500,
        # test_steps_per_task=100,
        train_transforms=[],
        test_transforms=[],
        val_transforms=[],
    )
    method = _DummyMethod()
    _ = setting.apply(method)
    # == 30 task switches in total.
    assert method.n_task_switches == 0
    assert method.n_fit_calls == 1
    assert not method.received_task_ids
    assert not method.received_while_training
