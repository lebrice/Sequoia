from typing import TypeVar, ClassVar

from sequoia.settings.assumptions.incremental_results import IncrementalResults
from sequoia.common.metrics.rl_metrics import EpisodeMetrics
MetricType = TypeVar("MetricsType", bound=EpisodeMetrics)


class DiscreteTaskAgnosticRLResults(IncrementalResults[MetricType]):
    """ Results for a whole train loop (transfer matrix), in an RL Setting.
    """
    # Higher mean reward / episode => better
    lower_is_better: ClassVar[bool] = False

    objective_name: ClassVar[str] = "Mean reward per episode"

    # Minimum runtime considered (in hours).
    # (No extra points are obtained for going faster than this.)
    min_runtime_hours: ClassVar[float] = 1.5
    # Maximum runtime allowed (in hours).
    max_runtime_hours: ClassVar[float] = 12.0
