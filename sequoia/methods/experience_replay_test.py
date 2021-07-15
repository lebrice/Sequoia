from sequoia.settings.sl import ClassIncrementalSetting, TaskIncrementalSLSetting
import pytest
from .experience_replay import ExperienceReplayMethod
from sequoia.common.config import Config
from sequoia.methods import Method
from sequoia.methods.method_test import MethodTests
from sequoia.conftest import slow
from sequoia.settings.sl import SLSetting
from typing import ClassVar, Type


class TestExperienceReplay(MethodTests):
    Method: ClassVar[Type[Method]] = ExperienceReplayMethod

    @classmethod
    @pytest.fixture
    def method(cls, config: Config) -> ExperienceReplayMethod:
        """ Fixture that returns the Method instance to use when testing/debugging.
        """
        return cls.Method()

    def validate_results(
        self,
        setting: SLSetting,
        method: ExperienceReplayMethod,
        results: SLSetting.Results,
    ) -> None:
        assert results
        assert results.objective

    @slow
    @pytest.mark.timeout(300)
    def test_class_incremental_mnist(self, config: Config):
        method = ExperienceReplayMethod(buffer_capacity=200, max_epochs_per_task=1)
        setting = ClassIncrementalSetting(
            dataset="mnist", monitor_training_performance=True,
        )
        results = setting.apply(method, config=config)
        assert 0.90 <= results.average_online_performance.objective

        assert 0.70 <= results.final_performance_metrics[0].objective
        assert 0.70 <= results.final_performance_metrics[1].objective
        assert 0.70 <= results.final_performance_metrics[2].objective
        assert 0.70 <= results.final_performance_metrics[3].objective
        assert 0.70 <= results.final_performance_metrics[4].objective

        assert 0.80 <= results.average_final_performance.objective
