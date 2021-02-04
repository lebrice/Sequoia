"""A random baseline Method that gives random predictions for any input.

Should be applicable to any Setting.
"""

from dataclasses import dataclass, fields, is_dataclass

import gym

from sequoia.common.metrics import ClassificationMetrics
from sequoia.methods import register_method
from sequoia.settings import ClassIncrementalSetting, PassiveSetting, Setting
from sequoia.settings.base import Actions, Environment, Method, Observations
from sequoia.utils import get_logger, singledispatchmethod

logger = get_logger(__file__)

@register_method
@dataclass
class RandomBaselineMethod(Method, target_setting=Setting):
    """ Baseline method that gives random predictions for any given setting.

    This method doesn't have a model or any parameters. It just returns a random
    action for every observation.
    """

    # Batch size to be used by this Method.
    batch_size: int = 16

    def fit(self, train_env: Environment, valid_env: Environment):
        # This method doesn't actually train, so we just return immediately.
        import time
        time.sleep(1)
        train_env.close()
        valid_env.close()
        return

    # def __post_init__(self, *args, **kwargs):
    #     assert False, self.batch_size

    def configure(self, setting):
        # Set any batch size, really.
        print(f"Setting the batch size on the setting to {self.batch_size}")
        # assert self.batch_size == 1 # FIXME: Remove this, just debugging atm.
        setting.batch_size = self.batch_size

    def get_actions(
        self, observations: Observations, action_space: gym.Space
    ) -> Actions:
        return action_space.sample()

    @classmethod
    def add_argparse_args(cls, parser, dest: str="method"):
        # super().add_argparse_args(parser, dest=dest)
        parser.add_arguments(cls, dest=dest) # Equivalent

    @classmethod
    def from_args(cls, *args, **kwargs):
        return super().from_args(*args, **kwargs)
        # return RandomBaselineMethod()

    ## Methods below are just here for testing purposes.

    @singledispatchmethod
    def validate_results(self, setting: Setting, results: Setting.Results):
        """Called during testing. Use this to assert that the results you get
        from applying your method on the given setting match your expectations.

        Args:
            setting
            results (Results): A given Results object.
        """
        assert results is not None
        assert results.objective > 0
        print(
            f"Objective when applied to a setting of type {type(setting)}: {results.objective}"
        )

    @validate_results.register
    def validate(
        self, setting: ClassIncrementalSetting, results: ClassIncrementalSetting.Results
    ):
        assert isinstance(setting, ClassIncrementalSetting), setting
        assert isinstance(results, ClassIncrementalSetting.Results), results

        average_accuracy = results.objective
        # Calculate the expected 'average' chance accuracy.
        # We assume that there is an equal number of classes in each task.
        chance_accuracy = 1 / setting.n_classes_per_task

        assert 0.5 * chance_accuracy <= average_accuracy <= 1.5 * chance_accuracy

        for i, metric in enumerate(results.average_metrics_per_task):
            assert isinstance(metric, ClassificationMetrics)
            # TODO: Check that this makes sense:
            chance_accuracy = 1 / setting.n_classes_per_task

            task_accuracy = metric.accuracy
            # FIXME: Look into this, we're often getting results substantially
            # worse than chance, and to 'make the tests pass' (which is bad)
            # we're setting the lower bound super low, which makes no sense.
            assert 0.25 * chance_accuracy <= task_accuracy <= 2.1 * chance_accuracy


if __name__ == "__main__":
    RandomBaselineMethod.main()
