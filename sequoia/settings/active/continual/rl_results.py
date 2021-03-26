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
from sequoia.common.metrics.rl_metrics import EpisodeMetrics


class RLResults(IncrementalResults[EpisodeMetrics]):
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

    @property
    def cl_score(self) -> float:
        """ CL Score, as a weigted sum of three objectives:
        - The average final performance over all tasks
        - The average 'online' performance over all tasks
        - Runtime

        TODO: @optimass Determine the weights for each factor.

        Returns
        -------
        float
            [description]
        """
        # TODO: Determine the function to use to get a runtime score between 0 and 1.
        # TODO: Add a property on the Results, which is based on the environment used.
        score = (
            +0.25 * self._online_performance_score()
            + 0.50 * self._final_performance_score()
            + 0.25 * self._runtime_score()
        )
        return score
