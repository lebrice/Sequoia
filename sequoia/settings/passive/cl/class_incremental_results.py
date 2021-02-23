""" Object representing the "Results" of applying a Method on a Class-Incremental Setting.

This object basically calculates the 'objective' specific to this setting as
well as provide a set of methods for making useful plots and utilities for
logging results to wandb.
""" 
from collections import defaultdict
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
from itertools import accumulate, chain
from functools import partial
from typing import Dict, List, Union, ClassVar
from pathlib import Path
import wandb

import matplotlib.pyplot as plt
from simple_parsing import field, list_field, mutable_field

from sequoia.common.loss import Loss
from sequoia.common.metrics import ClassificationMetrics, Metrics, RegressionMetrics
from sequoia.utils.logging_utils import get_logger
from sequoia.utils.plotting import PlotSectionLabel, autolabel
from sequoia.utils.utils import mean
from sequoia.settings.assumptions.incremental import IncrementalSetting

from .. import Results
logger = get_logger(__file__)


@dataclass
class ClassIncrementalResults(IncrementalSetting.Results):
    """Results for a ClassIncrementalSetting.
    
    The main objective in this setting is the average test accuracy over all
    tasks.
    
    The plots to generate are:
    - Accuracy per task
    - Average Test Accuray over the course of testing
    - Confusion matrix at the end of testing
    
    All of these will be created from the list of test metrics (Classification
    metrics for now).
    
    TODO: Add back Wandb logging somehow, even though we might be doing the
    evaluation loop ourselves.
    TODO: Fix this for the 'incremental regression' case.
    """
    test_metrics: List[List[Metrics]] = list_field(repr=False)
    objective_name: ClassVar[str] = "Average Accuracy"

    def make_plots(self):
        plots_dict = {}
        if wandb.run:
            # TODO: Add a Histogram plot from wandb?
            pass
        else:
            # TODO: Add back the plots.
            plots_dict["task_metrics"] = self.task_accuracies_plot()
        return plots_dict

    def task_accuracies_plot(self):
        figure: plt.Figure
        axes: plt.Axes
        figure, axes = plt.subplots()
        x = list(range(self.num_tasks))
        y = [metrics.accuracy for metrics in self.final_performance_metrics]
        rects = axes.bar(x, y)
        axes.set_title("Task Accuracy")
        axes.set_xlabel("Task")
        axes.set_ylabel("Accuracy")
        axes.set_ylim(0, 1.0)
        autolabel(axes, rects)
        return figure

    def cumul_metrics_plot(self):
        """TODO: Create a plot that shows the evolution of the test accuracy
        over all test tasks seen so far.

        (during training or during testing?)

        :return: [description]
        :rtype: [type]
        """
        figure: plt.Figure
        axes: plt.Axes
        figure, axes = plt.subplots()
        cumulative_metrics = list(accumulate(chain(*self.test_metrics)))

        if isinstance(cumulative_metrics[0], ClassificationMetrics):
            assert cumulative_metrics[-1].accuracy == self.average_metrics.accuracy

        x = [metrics.n_samples for metrics in cumulative_metrics]
        y = [metrics.accuracy for metrics in cumulative_metrics]
        axes.plot(x, y)
        return figure

    # def summary(self) -> str:
    #     s = StringIO()
    #     with redirect_stdout(s):
    #         for i, average_task_metrics in enumerate(self[-1].average_metrics_per_task):
    #             print(f"Test Results on task {i}: {average_task_metrics}")
    #         print(f"Average test metrics accross all the test tasks: {self[-1].average_metrics}")
    #     s.seek(0)
    #     return s.read()

    # def to_log_dict(self) -> Dict[str, float]:
    #     results = {}
    #     results[self.objective_name] = self.objective
    #     average_metrics = self[-1].average_metrics

    #     if isinstance(average_metrics, ClassificationMetrics):
    #         results["accuracy/average"] = average_metrics.accuracy
    #     elif isinstance(average_metrics, RegressionMetrics):
    #         results["mse/average"] = average_metrics.mse
    #     else:
    #         results["average metrics"] = average_metrics

    #     for i, average_task_metrics in enumerate(self[-1].average_metrics_per_task):
    #         if isinstance(average_task_metrics, ClassificationMetrics):
    #             results[f"accuracy/task_{i}"] = average_task_metrics.accuracy
    #         elif isinstance(average_task_metrics, RegressionMetrics):
    #             results[f"mse/task_{i}"] = average_task_metrics.mse
    #         else:
    #             results[f"task_{i}"] = average_task_metrics
    #     return results
