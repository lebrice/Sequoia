# TODO: Tests for the "traditional" RL setting.
from typing import ClassVar, Type

import pytest
from sequoia.settings import Setting

from ..incremental.setting_test import (
    TestIncrementalRLSetting as IncrementalRLSettingTests,
)
from .setting import TraditionalRLSetting
from sequoia.settings.rl.setting_test import DummyMethod
from sequoia.settings.assumptions.incremental_results import TaskSequenceResults


class TestTraditionalRLSetting(IncrementalRLSettingTests):
    Setting: ClassVar[Type[Setting]] = TraditionalRLSetting
    dataset: pytest.fixture

    def test_on_task_switch_is_called(self):
        setting = self.Setting(
            dataset="CartPole-v0",
            nb_tasks=5,
            # train_steps_per_task=100,
            train_max_steps=500,
            test_max_steps=500,
        )
        assert setting.stationary_context
        method = DummyMethod()
        _ = setting.apply(method)
        # assert setting.task_labels_at_test_time
        # assert False, method.observation_task_labels
        assert method.n_fit_calls == 1
        import numpy as np
        import torch

        assert torch.unique_consecutive(
            torch.as_tensor(method.observation_task_labels)
        ).tolist() != list(range(setting.nb_tasks))

    def validate_results(
        self,
        setting: TraditionalRLSetting,
        method: DummyMethod,
        results: TraditionalRLSetting.Results,
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
        assert not setting.task_labels_at_test_time
        if setting.nb_tasks == 1:
            assert not method.received_task_ids
            assert not method.received_while_training
        else:
            # Only received during testing.
            assert method.n_task_switches == t
            assert method.received_task_ids == [None for t_i in range(t)]
            assert method.received_while_training == [False for _ in range(t)]

    def validate_results(
        self,
        setting: TraditionalRLSetting,
        method: DummyMethod,
        results: TraditionalRLSetting.Results,
    ) -> None:
        assert results
        assert results.objective
        assert isinstance(results, TaskSequenceResults)
        assert len(results.task_results) == setting.nb_tasks
        assert results.average_metrics == sum(
            task_result.average_metrics for task_result in results.task_results
        )
        assert method.n_fit_calls == 1
        import numpy as np
        import torch

        train_task_labels = torch.as_tensor(method.observation_task_labels)
        new_train_task_labels = torch.unique_consecutive(train_task_labels).tolist()
        if setting.nb_tasks > 1:
            assert new_train_task_labels != list(range(setting.nb_tasks))
        else:
            assert set(method.observation_task_labels) == {0}
