from typing import ClassVar, Type

import pytest
from sequoia.common.config import Config
from sequoia.conftest import slow
from sequoia.methods.base_method import BaseMethod
from sequoia.methods.base_method_test import TestBaseMethod as BaseMethodTests
from sequoia.methods.method_test import MethodType, Setting
from sequoia.settings.sl import ClassIncrementalSetting, SLSetting

from .replay import Replay


class TestReplay(BaseMethodTests):
    Method: ClassVar[Type[BaseMethod]] = Replay

    @pytest.fixture()
    def method(self, config: Config) -> Replay:
        """Fixture that returns the Method instance to use when testing/debugging."""
        config.num_workers = 0
        return self.Method(config=config)

    def test_debug(self, method: MethodType, setting: Setting, config: Config):
        """Apply the Method onto a setting, and validate the results."""
        setting.num_workers = 0
        setting.drop_last = True
        results: Setting.Results = setting.apply(method, config=config)
        self.validate_results(setting=setting, method=method, results=results)

    def validate_results(
        self,
        setting: SLSetting,
        method: Replay,
        results: SLSetting.Results,
    ) -> None:
        assert results
        assert results.objective

    @slow
    @pytest.mark.timeout(300)
    def test_class_incremental_mnist(self, config: Config):
        method = self.Method(buffer_capacity=200, max_epochs_per_task=1)
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
