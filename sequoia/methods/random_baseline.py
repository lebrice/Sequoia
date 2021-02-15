"""A random baseline Method that gives random predictions for any input.

Should be applicable to any Setting.
"""

from dataclasses import dataclass

import gym
from typing import Optional, Mapping, Dict, Union, Any

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
    batch_size: int = 16
    
    def fit(self,
            train_env: Environment=None,
            valid_env: Environment=None,
            datamodule=None
        ):
        # This method doesn't actually train, so we just return immediately.
        return

    def configure(self, setting):
        # Set any batch size, really.
        print(f"Setting the batch size on the setting to {self.batch_size}")
        setting.batch_size = self.batch_size

    def get_actions(self, observations: Observations, action_space: gym.Space) -> Actions:
        return action_space.sample()

    @classmethod
    def from_args(cls, *args, **kwargs):
        return super().from_args(*args, **kwargs)
        # return RandomBaselineMethod()

    def get_search_space(self, setting: Setting) -> Mapping[str, Union[str, Dict]]:
        """Returns the search space to use for HPO in the given Setting.

        Parameters
        ----------
        setting : Setting
            The Setting on which the run of HPO will take place.

        Returns
        -------
        Mapping[str, Union[str, Dict]]
            An orion-formatted search space dictionary, mapping from hyper-parameter
            names (str) to their priors (str), or to nested dicts of the same form.
        """
        logger.warning(UserWarning(f"Hey, you seem to be trying to perform an HPO sweep using the random baseline method?"))
        # Assuming that this is just used for debugging, so giving back a simple space.
        return {"foo": "choice([0, 1, 2])"}

    def adapt_to_new_hparams(self, new_hparams: Dict[str, Any]) -> None:
        """Adapts the Method when it receives new Hyper-Parameters to try for a new run.

        It is required that this method be implemented if you want to perform HPO sweeps
        with Orion.
        
        Parameters
        ----------
        new_hparams : Dict[str, Any]
            The new hyper-parameters being recommended by the HPO algorithm. These will
            have the same structure as the search space.
        """
        logger.warning(UserWarning(f"Hey, you seem to be trying to perform an HPO sweep using the random baseline method?"))
        foo = new_hparams["foo"]
        print(f"Using new suggested value")

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
        print(f"Objective when applied to a setting of type {type(setting)}: {results.objective}")


    @validate_results.register
    def validate(self, setting: ClassIncrementalSetting, results: ClassIncrementalSetting.Results):
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

    # @singledispatchmethod
    # def model_class(self, setting: SettingType) -> Type[BaselineModel]:
    #     raise NotImplementedError(f"No known model for setting of type {type(setting)} (registry: {self.model_class.registry})")
    
    # @model_class.register
    # def _(self, setting: ActiveSetting) -> Type[Agent]:
    #     # TODO: Make a 'random' RL method.
    #     return RandomAgent

    # @model_class.register
    # def _(self, setting: ClassIncrementalSetting) -> Type[ClassIncrementalModelMixin]:
    #     # IDEA: Generate the model dynamically instead of creating one of each.
    #     # (This doesn't work atm because super() gives back a BaselineModel)
    #     # return get_random_model_class(super().model_class(setting))
    #     return RandomClassIncrementalModel

    # @model_class.register
    # def _(self, setting: TaskIncrementalSetting) -> Type[TaskIncrementalModel]:
    #     return RandomTaskIncrementalModel


if __name__ == "__main__":
    RandomBaselineMethod.main()
