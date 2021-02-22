import json
from io import StringIO
from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Dict, Union

import numpy as np
from simple_parsing import list_field

from sequoia.common import ClassificationMetrics, Metrics, RegressionMetrics
from sequoia.settings.assumptions.incremental import (
    IncrementalSetting,
    TaskResults,
    TaskSequenceResults,
    IncrementalResults,
)
from sequoia.settings.base import Results
from sequoia.utils import mean
from sequoia.utils.plotting import autolabel, plt
from sequoia.common.metrics.rl_metrics import EpisodeMetrics, RLMetrics


@dataclass
class RLResults(IncrementalResults[EpisodeMetrics]):
    """ Results for RL settings.

    Similar to ClassIncrementalResults, but here the metrics would be the mean
    reward, something like that.

    TODO: Add plots.
    """
    objective_name: ClassVar[str] = "Mean reward per episode"

    @property
    def objective(self) -> float:
        return self.mean_reward_per_episode

    @property
    def mean_reward(self):
        average_metric = self.average_metrics
        return average_metric.mse

    @property
    def mean_reward_per_episode(self) -> float:
        # Concatenate all the episodes from all tasks.
        return 

    @property
    def mean_episode_length(self) -> int:
        all_episode_lengths = sum(self.episode_lengths, [])
        return np.mean(all_episode_lengths)

    def to_log_dict(self, verbose: bool = False) -> Dict:
        log_dict = {}
        # TODO: This assumes that the metrics were stored in the right index for their
        # corresponding task.
        for task_id, task_sequence_result in enumerate(self):
            log_dict[f"Task {task_id}"] = task_sequence_result.to_log_dict(verbose=verbose)
        return log_dict

    def summary(self):
        s = StringIO()
        print(json.dumps(self.to_log_dict(), indent="\t"), file=s)
        s.seek(0)
        return s.read()

    def make_plots(self) -> Dict[str, Union[plt.Figure, Dict]]:
        return {
            f"Task {task_id}": task_sequence_result.make_plots()
            for task_id, task_sequence_result in enumerate(self)
        }

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
