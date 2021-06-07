# TODO: Tests for the multi-task RL setting.
from typing import ClassVar, Type

import pytest
from sequoia.settings import Setting

from ..task_incremental.setting_test import (
    TestTaskIncrementalRLSetting as TaskIncrementalRLSettingTests,
)
from ..task_incremental.setting_test import make_dataset_fixture, DummyMethod
from .setting import MultiTaskRLSetting


class TestMultiTaskRLSetting(TaskIncrementalRLSettingTests):
    Setting: ClassVar[Type[Setting]] = MultiTaskRLSetting
    dataset: pytest.fixture = make_dataset_fixture(MultiTaskRLSetting)

    def test_on_task_switch_is_called(self):
        setting = self.Setting(
            dataset="CartPole-v0",
            nb_tasks=5,
            # steps_per_task=100,
            train_max_steps=500,
            test_max_steps=500,
        )
        method = DummyMethod()
        _ = setting.apply(method)
        # 5 after learning task 0
        # 5 after learning task 1
        # 5 after learning task 2
        # 5 after learning task 3
        # 5 after learning task 4
        # == 30 task switches in total.
        assert method.n_task_switches == 0
        assert setting.task_labels_at_test_time