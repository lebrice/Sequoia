""" TODO: Create a Wrapper that measures performance over the first epoch of training in SL.

Then maybe after we can make something more general that also works for RL.
"""
import warnings
from abc import ABC
from collections import defaultdict
from typing import Dict, Generic, Iterable, List, Optional, Tuple

import numpy as np
from gym.utils import colorize
from torch import Tensor

from sequoia.common.gym_wrappers.utils import IterableWrapper
from sequoia.common.metrics import ClassificationMetrics, Metrics, MetricsType
from sequoia.common.metrics.rl_metrics import EpisodeMetrics
from sequoia.settings.base import (
    Actions,
    Environment,
    Observations,
    Rewards,
)
from sequoia.settings.passive.passive_environment import PassiveEnvironment


class MeasurePerformanceWrapper(IterableWrapper, Generic[MetricsType], ABC):
    def __init__(self, env: Environment):
        super().__init__(env)
        self.__metrics: Dict[int, MetricsType] = {}

    def get_online_performance(self) -> Dict[int, List[MetricsType]]:
        """Returns the online performance over the evaluation period.

        Returns
        -------
        Dict[int, MetricsType]
            A dict mapping from step number to the Metrics object captured at that step.
        """
        return dict(self.__metrics.copy())

    def get_average_online_performance(self) -> Optional[MetricsType]:
        """Returns the average online performance over the evaluation period, or None
        if the env was not iterated over / interacted with.

        Returns
        -------
        Optional[MetricsType]
            Metrics
        """
        if not self.__metrics:
            return None
        return sum(self._metrics.values())


class MeasureSLPerformanceWrapper(MeasurePerformanceWrapper[ClassificationMetrics]):
    def __init__(self, env: PassiveEnvironment, first_epoch_only: bool = False):
        super().__init__(env)
        self.__metrics: Dict[int, ClassificationMetrics] = defaultdict(int)
        self.first_epoch_only = first_epoch_only
        # Counter for the number of steps.
        self._steps: int = 0
        assert isinstance(self.env.unwrapped, PassiveEnvironment)
        if not self.env.unwrapped.pretend_to_be_active:
            # TODO: How do we prevent
            warnings.warn(
                RuntimeWarning(
                    colorize(
                        "Your performance on this environment will be monitored! "
                        "Since this env is Passive, i.e. a Supervised Learning "
                        "DataLoader, the Rewards (y) will be withheld until "
                        "actions are passed to the 'send' method. Make sure that "
                        "your training loop can handle this small tweak.",
                        color="yellow",
                    )
                )
            )
        self.env.unwrapped.pretend_to_be_active = True

    def reset(self) -> Observations:
        return self.env.reset()

    @property
    def in_evaluation_period(self) -> bool:
        if self.first_epoch_only:
            # TODO: Double-check the iteraction of IterableDataset and __len__
            return self._steps < self.env.__len__()
        return True

    def step(self, action: Actions):
        observation, reward, done, info = self.env.step(action)
        if self.in_evaluation_period:
            self.__metrics[self._steps] += self.get_metrics(action, reward)
        self._steps += 1
        return observation, reward, done, info

    def send(self, action: Actions):
        if not isinstance(action, Actions):
            assert isinstance(action, (np.ndarray, Tensor))
            action = Actions(action)
        reward = self.env.send(action)
        if self.in_evaluation_period:
            self.__metrics[self._steps] += self.get_metrics(action, reward)
        # This is ok since we don't increment in the iterator.
        self._steps += 1
        return reward

    def get_metrics(self, action: Actions, reward: Rewards) -> Metrics:
        assert action.y_pred.shape == reward.y.shape, (action.shapes, reward.shapes)
        return ClassificationMetrics(
            y_pred=action.y_pred, y=reward.y, num_classes=self.n_classes
        )

    def __iter__(self) -> Iterable[Tuple[Observations, Optional[Rewards]]]:
        for obs, _ in self.env.__iter__():
            self._observation = obs
            yield obs, None


class MeasureRLPermanceWrapper(MeasurePerformanceWrapper[EpisodeMetrics]):
    # TODO
    pass
