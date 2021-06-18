from typing import TypeVar, ClassVar

from sequoia.settings.assumptions.discrete_results import TaskSequenceResults
from sequoia.common.metrics.rl_metrics import EpisodeMetrics
MetricType = TypeVar("MetricsType", bound=EpisodeMetrics)


class DiscreteTaskAgnosticRLResults(TaskSequenceResults[MetricType]):
    """ Results for a sequence of tasks in an RL Setting
    
    This can be seen as one row of a transfer matrix.
    NOTE: This is not the entire transfer matrix because in the Discrete settings we don't
    evaluate after learning each task.
    """
    # Higher mean reward / episode => better
    lower_is_better: ClassVar[bool] = False

    objective_name: ClassVar[str] = "Mean reward per episode"

    # Minimum runtime considered (in hours).
    # (No extra points are obtained for going faster than this.)
    min_runtime_hours: ClassVar[float] = 1.5
    # Maximum runtime allowed (in hours).
    max_runtime_hours: ClassVar[float] = 12.0
