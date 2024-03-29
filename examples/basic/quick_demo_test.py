""" TODO: Write tests that check that the examples are working correctly.
"""
import contextlib
import sys

import pytest

from examples.basic.quick_demo import demo_command_line, demo_simple
from sequoia.settings import ClassIncrementalSetting, Results


@pytest.mark.timeout(120)
def test_quick_demo(monkeypatch):
    """Test that runs the quick demo and checks that the results correspond to
    what you'd expect.
    """
    results: ClassIncrementalSetting.Results = None
    summary_method = ClassIncrementalSetting.Results.summary

    def summary(self: ClassIncrementalSetting.Results):
        nonlocal results
        results = self
        return summary_method(self)

    monkeypatch.setattr(ClassIncrementalSetting.Results, "summary", summary)

    demo_simple()

    from sequoia.common.metrics import ClassificationMetrics

    # NOTE: Results aren't going to give *exactly* the same results, so we can't
    # test like this directly:
    # assert results.average_metrics_per_task == [
    #     ClassificationMetrics(n_samples=1984, accuracy=0.500504),
    #     ClassificationMetrics(n_samples=2016, accuracy=0.499504),
    #     ClassificationMetrics(n_samples=1984, accuracy=0.817036),
    #     ClassificationMetrics(n_samples=2016, accuracy=0.835317),
    #     ClassificationMetrics(n_samples=1984, accuracy=0.99748),
    # ]

    assert results.final_performance_metrics[0].n_samples == 1984
    assert results.final_performance_metrics[1].n_samples == 2016
    assert results.final_performance_metrics[2].n_samples == 1984
    assert results.final_performance_metrics[3].n_samples == 2016
    assert results.final_performance_metrics[4].n_samples == 1984

    assert 0.48 <= results.final_performance_metrics[0].accuracy <= 0.55
    assert 0.48 <= results.final_performance_metrics[1].accuracy <= 0.70
    assert 0.60 <= results.final_performance_metrics[2].accuracy <= 1.00
    assert 0.70 <= results.final_performance_metrics[3].accuracy <= 1.00
    assert 0.99 <= results.final_performance_metrics[4].accuracy <= 1.00
