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


@dataclass
class RLResults(IncrementalResults[EpisodeMetrics]):
    """ Results for a whole train loop (transfer matrix), in an RL Setting.
    """
    objective_name: ClassVar[str] = "Mean reward per episode"

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
