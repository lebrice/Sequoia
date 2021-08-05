""" TODO: Create a Wrapper that measures performance over the first epoch of training in SL.

Then maybe after we can make something more general that also works for RL.
"""
import warnings
from abc import ABC
from collections import defaultdict

""" Wrapper that gets applied onto the environment in order to measure the online
training performance.

TODO: Move this somewhere more appropriate. There's also the RL version of the wrapper
here.
"""
from typing import Any, Dict, Generic, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import wandb
from gym.utils import colorize
from gym.vector import VectorEnv
from sequoia.common.gym_wrappers.measure_performance import MeasurePerformanceWrapper
from sequoia.common.gym_wrappers.utils import EnvType, IterableWrapper
from sequoia.common.metrics import ClassificationMetrics, Metrics, MetricsType
from sequoia.common.metrics.rl_metrics import EpisodeMetrics
from sequoia.settings.base import Actions, Environment, Observations, Rewards
from sequoia.settings.sl.environment import PassiveEnvironment
from sequoia.utils.utils import add_prefix
from torch import Tensor
from sequoia.common.gym_wrappers.batch_env.tile_images import tile_images


class MeasureSLPerformanceWrapper(
    MeasurePerformanceWrapper,
    # MeasurePerformanceWrapper[PassiveEnvironment]  # Python 3.7
    # MeasurePerformanceWrapper[PassiveEnvironment, ClassificationMetrics] # Python 3.8+
):
    def __init__(
        self,
        env: PassiveEnvironment,
        first_epoch_only: bool = False,
        wandb_prefix: str = None,
    ):
        super().__init__(env)
        # Metrics mapping from step to the metrics at that step.
        self._metrics: Dict[int, ClassificationMetrics] = defaultdict(Metrics)
        self.first_epoch_only = first_epoch_only
        self.wandb_prefix = wandb_prefix
        # Counter for the number of steps.
        self._steps: int = 0
        assert isinstance(self.env.unwrapped, PassiveEnvironment)
        if not self.env.unwrapped.pretend_to_be_active:
            warnings.warn(
                RuntimeWarning(
                    colorize(
                        "Your online performance "
                        + ("during the first epoch " if self.first_epoch_only else "")
                        + "on this environment will be monitored! "
                        "Since this env is Passive, i.e. a Supervised Learning "
                        "DataLoader, the Rewards (y) will be withheld until "
                        "actions are passed to the 'send' method. Make sure that "
                        "your training loop can handle this small tweak.",
                        color="yellow",
                    )
                )
            )
        self.env.unwrapped.pretend_to_be_active = True
        self.__epochs = 0

    def reset(self) -> Observations:
        return self.env.reset()

    @property
    def in_evaluation_period(self) -> bool:
        if self.first_epoch_only:
            # TODO: Double-check the iteraction of IterableDataset and __len__
            return self.__epochs == 0
        return True

    def step(self, action: Actions):
        observation, reward, done, info = self.env.step(action)
        # TODO: Make this wrapper task-aware, using the task ids in this `observation`?
        if self.in_evaluation_period:
            # TODO: Edge case, but we also need the prediction for the last batch to be
            # counted.
            self._metrics[self._steps] += self.get_metrics(action, reward)
        elif self.first_epoch_only:
            # If we are at the last batch in the first epoch, we still keep the metrics
            # for that batch, even though we're technically not in the first epoch
            # anymore.
            # TODO: CHeck the length through the dataset? or through a more 'clean' way
            # e.g. through the `max_steps` property of a TimeLimit wrapper or something?
            num_batches = len(self.unwrapped.dataset) // self.batch_size
            if not self.unwrapped.drop_last:
                num_batches += 1 if len(self.unwrapped.dataset) % self.batch_size else 0
            # currently_at_last_batch = self._steps == num_batches - 1
            currently_at_last_batch = self._steps == num_batches - 1
            if self.__epochs == 1 and currently_at_last_batch:
                self._metrics[self._steps] += self.get_metrics(action, reward)
        self._steps += 1
        return observation, reward, done, info

    def send(self, action: Actions):
        if not isinstance(action, Actions):
            assert isinstance(action, (np.ndarray, Tensor))
            action = Actions(action)

        reward = self.env.send(action)

        if self.in_evaluation_period:
            # TODO: Edge case, but we also need the prediction for the last batch to be
            # counted.
            self._metrics[self._steps] += self.get_metrics(action, reward)
        elif self.first_epoch_only:
            # If we are at the last batch in the first epoch, we still keep the metrics
            # for that batch, even though we're technically not in the first epoch
            # anymore.
            # TODO: CHeck the length through the dataset? or through a more 'clean' way
            # e.g. through the `max_steps` property of a TimeLimit wrapper or something?
            num_batches = len(self.unwrapped.dataset) // self.batch_size
            if not self.unwrapped.drop_last:
                num_batches += 1 if len(self.unwrapped.dataset) % self.batch_size else 0
            # currently_at_last_batch = self._steps == num_batches - 1
            currently_at_last_batch = self._steps == num_batches - 1
            if self.__epochs == 1 and currently_at_last_batch:
                self._metrics[self._steps] += self.get_metrics(action, reward)
        # This is ok since we don't increment in the iterator.
        self._steps += 1
        return reward

    def get_metrics(self, action: Actions, reward: Rewards) -> Metrics:
        assert action.y_pred.shape == reward.y.shape, (action.shapes, reward.shapes)
        metric = ClassificationMetrics(
            y_pred=action.y_pred, y=reward.y, num_classes=self.n_classes
        )

        if wandb.run:
            log_dict = metric.to_log_dict()
            if self.wandb_prefix:
                log_dict = add_prefix(log_dict, prefix=self.wandb_prefix, sep="/")
            log_dict["steps"] = self._steps
            wandb.log(log_dict)
        return metric

    def __iter__(self) -> Iterator[Tuple[Observations, Optional[Rewards]]]:
        if self.__epochs == 1 and self.first_epoch_only:
            print(
                colorize(
                    "Your performance during the first epoch on this environment has "
                    "been successfully measured! The environment will now yield the "
                    "rewards (y) during iteration, and you are no longer required to "
                    "send an action for each observation.",
                    color="green",
                )
            )
            self.env.unwrapped.pretend_to_be_active = False

        for obs, rew in self.env.__iter__():
            if self.in_evaluation_period:
                yield obs, None
            else:
                yield obs, rew
        self.__epochs += 1

