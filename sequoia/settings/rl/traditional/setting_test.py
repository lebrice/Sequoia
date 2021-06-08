# TODO: Tests for the "traditional" RL setting.
from typing import ClassVar, Type

import pytest
from sequoia.settings import Setting

from ..incremental.setting_test import (
    TestIncrementalRLSetting as IncrementalRLSettingTests,
)
from ..incremental.setting_test import make_dataset_fixture
from .setting import TraditionalRLSetting
from sequoia.settings.rl.setting_test import DummyMethod
from sequoia.settings.assumptions.incremental_results import TaskSequenceResults


def test_on_task_switch_is_called():
    setting = TraditionalRLSetting(
        dataset="CartPole-v0",
        nb_tasks=5,
        # steps_per_task=100,
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
    assert torch.unique_consecutive(torch.as_tensor(method.observation_task_labels)).tolist() != list(range(setting.nb_tasks))


class TestTraditionalRLSetting(IncrementalRLSettingTests):
    Setting: ClassVar[Type[Setting]] = TraditionalRLSetting
    dataset: pytest.fixture = make_dataset_fixture(TraditionalRLSetting)

    
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
        assert results.average_metrics == sum(task_result.average_metrics for task_result in results.task_results)
        assert method.n_fit_calls == 1
        import numpy as np
        import torch

        train_task_labels = torch.as_tensor(method.observation_task_labels)
        new_train_task_labels = torch.unique_consecutive(train_task_labels).tolist()
        if setting.nb_tasks > 1:
            assert new_train_task_labels != list(range(setting.nb_tasks))
        else:
            assert set(method.observation_task_labels) == {0}
