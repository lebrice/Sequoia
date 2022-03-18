# TODO: Tests for the multi-task RL setting.
from typing import ClassVar, Type

import pytest

from sequoia.settings.rl.setting_test import DummyMethod

from ..task_incremental.setting_test import (
    TestTaskIncrementalRLSetting as TaskIncrementalRLSettingTests,
)
from .setting import MultiTaskRLSetting


class TestMultiTaskRLSetting(TaskIncrementalRLSettingTests):
    Setting: ClassVar[Type[Setting]] = MultiTaskRLSetting
    dataset: pytest.fixture

    # def test_on_task_switch_is_called(self):
    #     setting = self.Setting(
    #         dataset="CartPole-v0",
    #         nb_tasks=5,
    #         # train_steps_per_task=100,
    #         train_max_steps=500,
    #         test_max_steps=500,
    #     )
    #     method = DummyMethod()
    #     _ = setting.apply(method)
    #     assert setting.task_labels_at_test_time
    #     assert False, method.observation_task_labels

    def validate_results(
        self,
        setting: MultiTaskRLSetting,
        method: DummyMethod,
        results: MultiTaskRLSetting.Results,
    ) -> None:
        """Check that the results make sense.
        The Dummy Method used also keeps useful attributes, which we check here.
        """
        assert results
        assert results.objective
        assert setting.stationary_context
        assert len(results.task_results) == setting.nb_tasks
        assert results.average_metrics == sum(
            task_result.average_metrics for task_result in results.task_results
        )
        t = setting.nb_tasks
        p = setting.phases
        assert setting.known_task_boundaries_at_train_time
        assert setting.known_task_boundaries_at_test_time
        assert setting.task_labels_at_train_time
        assert setting.task_labels_at_test_time
        if setting.nb_tasks == 1:
            assert not method.received_task_ids
            assert not method.received_while_training
        else:
            # Only received during testing.
            assert method.received_task_ids == [t_i for t_i in range(t)]
            assert method.received_while_training == [False for _ in range(t)]
