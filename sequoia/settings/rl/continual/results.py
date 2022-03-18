from typing import ClassVar, Generic, TypeVar

from sequoia.common.metrics.rl_metrics import EpisodeMetrics
from sequoia.settings.assumptions.continual import ContinualResults
from sequoia.utils.plotting import autolabel, plt

MetricType = TypeVar("MetricType", bound=EpisodeMetrics)


class ContinualRLResults(ContinualResults, Generic[MetricType]):
    """Results for a ContinualRLSetting."""

    # Higher mean reward / episode => better
    lower_is_better: ClassVar[bool] = False

    objective_name: ClassVar[str] = "Mean reward per episode"

    # Minimum runtime considered (in hours).
    # (No extra points are obtained for going faster than this.)
    min_runtime_hours: ClassVar[float] = 1.5
    # Maximum runtime allowed (in hours).
    max_runtime_hours: ClassVar[float] = 12.0

    def mean_reward_plot(self):
        raise NotImplementedError("TODO")
        figure: plt.Figure
        axes: plt.Axes
        figure, axes = plt.subplots()
        x = list(range(self.num_tasks))
        y = [metrics.accuracy for metrics in self.average_metrics_per_task]
        rects = axes.bar(x, y)
        axes.set_title("Task Accuracy")
        axes.set_xlabel("Task")
        axes.set_ylabel("Accuracy")
        axes.set_ylim(0, 1.0)
        autolabel(axes, rects)
        return figure
