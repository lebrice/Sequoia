""" Object representing the "Results" of applying a Method on a Class-Incremental Setting.

This object basically calculates the 'objective' specific to this setting as
well as provide a set of methods for making useful plots and utilities for
logging results to wandb.
"""
from typing import ClassVar

import matplotlib.pyplot as plt

import wandb
from sequoia.settings.assumptions.incremental import IncrementalAssumption
from sequoia.utils.logging_utils import get_logger
from sequoia.utils.plotting import autolabel

logger = get_logger(__file__)


class IncrementalSLResults(IncrementalAssumption.Results):
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

    # Higher accuracy => better
    lower_is_better: ClassVar[bool] = False
    objective_name: ClassVar[str] = "Average Accuracy"

    # Minimum runtime considered (in hours).
    # (No extra points are obtained when going faster than this.)
    min_runtime_hours: ClassVar[float] = 5.0 / 60.0  # 5 minutes
    # Maximum runtime allowed (in hours).
    max_runtime_hours: ClassVar[float] = 1.0  # one hour.

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
        """TODO: Create a plot that shows the evolution of the test performance over
        all test tasks seen so far.

        (during training or during testing?)
        """
        figure: plt.Figure
        axes: plt.Axes
        figure, axes = plt.subplots()
        x = list(range(self.num_tasks))
        y = []
        metric_name: str = ""
        for i in range(self.num_tasks):
            previous_metrics = self.metrics_matrix[i][: i + 1]
            cumul_metrics = sum(previous_metrics)
            y.append(cumul_metrics.objective)
            if not metric_name:
                metric_name = cumul_metrics.objective_name

        # x = [metrics.n_samples for metrics in cumulative_metrics]
        # y = [metrics.accuracy for metrics in cumulative_metrics]
        axes.plot(x, y)
        axes.set_xlabel("# of learned tasks")
        axes.set_ylabel(f"Average {metric_name} on tasks seen so far")
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
