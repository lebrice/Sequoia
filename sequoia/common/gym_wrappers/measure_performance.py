""" Abstract base class for a Wrapper that gets applied onto the environment in order to
measure the online training performance.

The concrete versions of this wrapper are located.
"""
from abc import ABC
from typing import Dict, Generic, List, Optional

from sequoia.common.gym_wrappers.utils import EnvType, IterableWrapper
from sequoia.common.metrics import MetricsType
from sequoia.settings.base import Environment


class MeasurePerformanceWrapper(IterableWrapper[EnvType], Generic[EnvType, MetricsType], ABC):
    def __init__(self, env: Environment):
        super().__init__(env)
        self._metrics: Dict[int, MetricsType] = {}

    def get_online_performance(self) -> Dict[int, List[MetricsType]]:
        """Returns the online performance over the evaluation period.

        Returns
        -------
        Dict[int, MetricsType]
            A dict mapping from step number to the Metrics object captured at that step.
        """
        return dict(self._metrics.copy())

    def get_average_online_performance(self) -> Optional[MetricsType]:
        """Returns the average online performance over the evaluation period, or None
        if the env was not iterated over / interacted with.

        Returns
        -------
        Optional[MetricsType]
            Metrics
        """
        if not self._metrics:
            return None
        return sum(self._metrics.values())
