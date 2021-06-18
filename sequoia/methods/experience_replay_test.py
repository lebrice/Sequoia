from sequoia.settings.sl import ClassIncrementalSetting, TaskIncrementalSLSetting
import pytest
from .experience_replay import ExperienceReplayMethod
from sequoia.common.config import Config


@pytest.mark.timeout(300)
def test_class_incremental_mnist(config: Config):
    method = ExperienceReplayMethod(buffer_capacity=200, max_epochs_per_task=1)
    setting = ClassIncrementalSetting(
        dataset="mnist",
        monitor_training_performance=True,
    )
    results = setting.apply(method, config=config)
    assert 0.90 <= results.average_online_performance.objective

    assert 0.70 <= results.final_performance_metrics[0].objective
    assert 0.70 <= results.final_performance_metrics[1].objective
    assert 0.70 <= results.final_performance_metrics[2].objective
    assert 0.70 <= results.final_performance_metrics[3].objective
    assert 0.70 <= results.final_performance_metrics[4].objective

    assert 0.80 <= results.average_final_performance.objective
