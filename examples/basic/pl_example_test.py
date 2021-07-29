""" Unit-tests for the PyTorch-Lightning Example.

Can be run like so:
```console
$ pytest examples/basic/pl_example_test.py
```
"""
import pytest
from typing import Type

from sequoia.common.config import Config
from sequoia.common.metrics import ClassificationMetrics
from sequoia.methods import Method
from sequoia.methods.method_test import MethodTests, config, session_config  # type: ignore
from sequoia.settings import Results
from sequoia.settings.sl import (
    ContinualSLSetting,
    IncrementalSLSetting,
)
from examples.basic.pl_example import ExampleMethod, Model


class TestPLExample(MethodTests):
    """ Tests for this PL Example.

    This `MethodTests` base class generates a `test_debug` test for us.
    """

    Method: Type[Method] = ExampleMethod

    @pytest.fixture()
    def method(self, config: Config):
        """ Required fixture, which creates a Method that can be used for quick tests.
        """
        return ExampleMethod(hparams=Model.HParams(max_epochs_per_task=1))

    def validate_results(
        self, setting: ContinualSLSetting, method: ExampleMethod, results: Results
    ):
        """ This gets called by `test_debug` to check that the results make sense for
        the given setting and method.

        """
        # NOTE: This particular example isn't that great: We just check that the average
        # final test accuracy and the average online accuracy are both non-zero.
        # It would be best to do some kind of branching depending on what type of
        # Setting was used, since each setting can produce different types of results.
        print(results.summary())

        average_metrics: ClassificationMetrics
        online_metrics: ClassificationMetrics

        assert setting.monitor_training_performance

        todo = 0.0
        if isinstance(setting, IncrementalSLSetting):
            # The results in this case include the entire nb_tasks x nb_tasks transfer
            # matrix.
            assert isinstance(results, IncrementalSLSetting.Results)
            average_metrics = results.average_final_performance
            online_metrics = results.average_online_performance

            if setting.stationary_context:
                # Example: Should expect better performance if the data is i.i.d!
                assert average_metrics.accuracy > todo
            else:
                assert average_metrics.accuracy > todo

            if setting.monitor_training_performance:
                assert online_metrics.accuracy > todo
        else:
            # In this case, there aren't clear 'tasks' to speak of, so the results are
            # just aggregated metrics for each test batch:
            assert isinstance(results, ContinualSLSetting.Results)
            average_metrics = results.average_metrics
            online_metrics = results.online_performance_metrics

            assert average_metrics.accuracy > todo
            assert online_metrics.accuracy > todo
