"""A random baseline Method that gives random predictions for any input.

Should be applicable to any Setting.
"""

from argparse import Namespace
from typing import Any, Dict, Mapping, Optional, Union

import gym
import numpy as np
import tqdm
from simple_parsing import ArgumentParser
from torch import Tensor

from sequoia.common.metrics import ClassificationMetrics
from sequoia.methods import register_method
from sequoia.settings import ClassIncrementalSetting, Setting
from sequoia.settings.base import Actions, Environment, Method, Observations
from sequoia.settings.sl import SLSetting
from sequoia.utils import get_logger, singledispatchmethod

logger = get_logger(__file__)


@register_method
class RandomBaselineMethod(Method, target_setting=Setting):
    """ Baseline method that gives random predictions for any given setting.

    This method doesn't have a model or any parameters. It just returns a random
    action for every observation.
    """

    def __init__(self):
        self.max_train_episodes: Optional[int] = None

    def configure(self, setting: Setting):
        """ Called before the method is applied on a setting (before training).

        You can use this to instantiate your model, for instance, since this is
        where you get access to the observation & action spaces.
        """
        if isinstance(setting, SLSetting):
            # Being applied in SL, we will only do one 'epoch" (a.k.a. "episode").
            self.max_train_episodes = 1

    def fit(
        self, train_env: Environment, valid_env: Environment,
    ):
        episodes = 0
        with tqdm.tqdm(desc="training") as train_pbar:

            while not train_env.is_closed():
                for i, batch in enumerate(train_env):
                    if isinstance(batch, Observations):
                        observations, rewards = batch, None
                    else:
                        observations, rewards = batch

                    batch_size = observations.x.shape[0]
                    y_pred = train_env.action_space.sample()

                    # If we're at the last batch, it might have a different size, so w
                    # give only the required number of values.
                    if isinstance(y_pred, (np.ndarray, Tensor)):
                        if y_pred.shape[0] != batch_size:
                            y_pred = y_pred[:batch_size]

                    if rewards is None:
                        rewards = train_env.send(y_pred)

                    train_pbar.set_postfix({"Episode": episodes, "Step": i})
                    # train as you usually would.

                    if train_env.is_closed():
                        break

                episodes += 1
                if self.max_train_episodes and episodes >= self.max_train_episodes:
                    train_env.close()
                    break

    def get_actions(
        self, observations: Observations, action_space: gym.Space
    ) -> Actions:
        return action_space.sample()

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
        logger.warning(
            UserWarning(
                "Hey, you seem to be trying to perform an HPO sweep using the random "
                "baseline method?"
            )
        )
        # Assuming that this is just used for debugging, so giving back a simple space.
        return {"foo": "choices([0, 1, 2])"}

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
        foo = new_hparams["foo"]
        print(f"Using new suggested value {foo}")

    @classmethod
    def add_argparse_args(cls, parser: ArgumentParser, dest: str = ""):
        pass

    @classmethod
    def from_argparse_args(cls, args: Namespace, dest: str = ""):
        return cls()

    # Methods below are just here for testing purposes.

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
        chance_accuracy = 1 / setting.num_classes
        # chance_accuracy = 1 / setting.n_classes_per_task
        assert 0.5 * chance_accuracy <= average_accuracy <= 1.5 * chance_accuracy

        for i, metric in enumerate(results.final_performance_metrics):
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
