""" Object representing the "Results" of applying a Method on a Class-Incremental Setting.

This object basically calculates the 'objective' specific to this setting as
well as provide a set of methods for making useful plots and utilities for
logging results to wandb.
""" 
from collections import OrderedDict, defaultdict
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
from itertools import accumulate, chain
from functools import partial
from typing import Dict, List, Union
from pathlib import Path


import matplotlib.pyplot as plt
from simple_parsing import field, list_field, mutable_field

from common.loss import Loss
from common.metrics import ClassificationMetrics, Metrics, RegressionMetrics
from utils.logging_utils import get_logger
from utils.plotting import PlotSectionLabel, autolabel
from utils.utils import mean

from .. import Results
from settings.assumptions.incremental import IncrementalSetting
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
    
    TODO: Add back Wandb logging somehow, even though we're doing the
    evaluation loop ourselves.
    """
    test_metrics: List[List[Metrics]] = list_field(repr=False)

    def save_to_dir(self, save_dir: Union[str, Path]) -> None:
        # TODO: Add wandb logging here somehow.
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        plots: Dict[str, plt.Figure] = self.make_plots()
        print(f"\nPlots: {plots}\n")

        results_json_path = save_dir / "results.json"
        self.save(results_json_path)
        print(f"Saved a copy of the results to {results_json_path}")

        for fig_name, figure in plots.items():
            print(f"fig_name: {fig_name}")
            figure.show()
            plt.waitforbuttonpress(10)
            path = (save_dir/ fig_name).with_suffix(".jpg")
            path.parent.mkdir(exist_ok=True, parents=True)
            figure.savefig(path)
            print(f"Saved figure at path {path}")

    def make_plots(self):
        results = {
            "task_accuracies": self.task_accuracies_plot()
        }
        return results

    def task_accuracies_plot(self):
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

    def summary(self) -> str:
        s = StringIO()
        with redirect_stdout(s):
            for i, average_task_metrics in enumerate(self.average_metrics_per_task):
                print(f"Test Results on task {i}: {average_task_metrics}")
            print(f"Average test metrics accross all the test tasks: {self.average_metrics}")
        s.seek(0)
        return s.read()

    def to_log_dict(self) -> Dict[str, float]:
        results = OrderedDict()
        results["objective"] = self.objective
        for i, average_task_metrics in enumerate(self.average_metrics_per_task):
            if isinstance(average_task_metrics, ClassificationMetrics):
                results[f"task_{i}/accuracy"] = average_task_metrics.accuracy
            elif isinstance(average_task_metrics, RegressionMetrics):
                results[f"task_{i}/mse"] = average_task_metrics.mse
            else:
                results[f"task_{i}"] = average_task_metrics
        return results
