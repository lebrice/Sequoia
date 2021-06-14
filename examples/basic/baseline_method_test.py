""" TODO: "Integration" tests for the BaselineMethod. """

import pytest
from sequoia.common import Config
from sequoia.conftest import slow
from sequoia.methods import BaselineMethod
from sequoia.settings import (ClassIncrementalSetting,
                              TraditionalSLSetting,
                              TaskIncrementalSLSetting)
from sequoia.settings.sl import MultiTaskSLSetting


@pytest.mark.xfail(reason="WIP")
@pytest.mark.timeout(120)
def test_class_incremental_setting():
    method = BaselineMethod(no_wandb=True, max_epochs=1)
    setting = ClassIncrementalSetting()
    results = setting.apply(method)
    print(results.summary())

    assert results.final_performance_metrics[0].n_samples == 1984
    assert results.final_performance_metrics[1].n_samples == 2016
    assert results.final_performance_metrics[2].n_samples == 1984
    assert results.final_performance_metrics[3].n_samples == 2016
    assert results.final_performance_metrics[4].n_samples == 1984

    assert 0.48 <= results.final_performance_metrics[0].accuracy <= 0.55
    assert 0.48 <= results.final_performance_metrics[1].accuracy <= 0.55
    assert 0.60 <= results.final_performance_metrics[2].accuracy <= 0.95
    assert 0.75 <= results.final_performance_metrics[3].accuracy <= 0.98
    assert 0.99 <= results.final_performance_metrics[4].accuracy <= 1.00




@pytest.mark.timeout(300)
def test_multi_task_setting():
    method = BaselineMethod(no_wandb=True, max_epochs=1)
    setting = MultiTaskSLSetting(dataset="mnist")
    results = setting.apply(method)
    print(results.summary())

    assert results.final_performance_metrics[0].n_samples == 2112
    assert results.final_performance_metrics[1].n_samples == 2016
    assert results.final_performance_metrics[2].n_samples == 1888
    assert results.final_performance_metrics[3].n_samples == 1984
    assert results.final_performance_metrics[4].n_samples == 1984

    assert 0.95 <= results.final_performance_metrics[0].accuracy
    assert 0.95 <= results.final_performance_metrics[1].accuracy
    assert 0.95 <= results.final_performance_metrics[2].accuracy
    assert 0.95 <= results.final_performance_metrics[3].accuracy
    assert 0.95 <= results.final_performance_metrics[4].accuracy

