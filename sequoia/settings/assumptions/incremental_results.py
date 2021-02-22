""" Results of an Incremental setting. """
import json
from io import StringIO
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import wandb
from sequoia.common.metrics import Metrics

from .iid_results import MetricType, TaskResults


class TaskSequenceResults(List[TaskResults[MetricType]]):
    """ Results for a sequence of Tasks. """

    @property
    def num_tasks(self) -> int:
        """Returns the number of tasks.

        Returns
        -------
        int
            Number of tasks.
        """
        return len(self)

    @property
    def average_metrics(self) -> MetricType:
        return sum(self.average_metrics_per_task, Metrics())

    @property
    def average_metrics_per_task(self) -> List[MetricType]:
        return [task_result.average_metrics for task_result in self]

    @property
    def objective(self) -> float:
        return self.average_metrics.objective

    def to_log_dict(self, verbose: bool = False) -> Dict:
        result = {}
        for task_id, task_results in enumerate(self):
            result[f"Task {task_id}"] = task_results.to_log_dict(verbose=verbose)
        return result

    def make_plots(self) -> Dict[str, plt.Figure]:
        result = {}
        for task_id, task_results in enumerate(self):
            result[f"Task {task_id}"] = task_results.make_plots()
        return result


class IncrementalResults(List[TaskSequenceResults[MetricType]]):
    """ Results for a whole train loop (transfer matrix). """

    @property
    def transfer_matrix(self) -> List[List[TaskResults]]:
        return self

    @property
    def objective_matrix(self) -> List[List[float]]:
        """Return transfer matrix containing the value of the 'objective' for each task.

        The value at the index (i, j) gives the test performance on task j after having
        learned tasks 0-i.

        Returns
        -------
        List[List[float]]
            The 2d matrix of objectives (floats).
        """
        return [
            [task_result.objective for task_result in task_sequence_result]
            for task_sequence_result in self.transfer_matrix
        ]


    @property
    def num_tasks(self) -> int:
        return len(self)

    @property
    def online_performance(self) -> List[TaskResults[MetricType]]:
        """ Returns the "online" performance, i.e. the diagonal of the transfer matrix.
        
        NOTE: This doesn't exactly show online performance, since testing happens after
        training is done on each task. This shows the performance on the most recently
        learned task, over the course of training.
        """
        return [self[i][i] for i in range(self.num_tasks)]

    @property
    def final_performance(self) -> List[TaskResults[MetricType]]:
        return self[-1]

    @property
    def avereage_online_performance(self) -> MetricType:
        return sum(self.online_performance, Metrics())

    @property
    def average_final_performance(self) -> MetricType:
        return sum(self.final_performance, Metrics())

    def to_log_dict(self, verbose: bool = False) -> Dict:
        log_dict = {}
        # TODO: This assumes that the metrics were stored in the right index for their
        # corresponding task.
        for task_id, task_sequence_result in enumerate(self):
            log_dict[f"Task {task_id}"] = task_sequence_result.to_log_dict(
                verbose=verbose
            )
        return log_dict

    def summary(self):
        s = StringIO()
        print(json.dumps(self.to_log_dict(), indent="\t"), file=s)
        s.seek(0)
        return s.read()

    def make_plots(self) -> Dict[str, Union[plt.Figure, Dict]]:
        plots = {
            f"Task {task_id}": task_sequence_result.make_plots()
            for task_id, task_sequence_result in enumerate(self)
        }
        axis_labels = [f"Task {task_id}" for task_id in range(len(self))]
        if wandb.run:
            plots["Transfer matrix"] = wandb.plots.HeatMap(
                x_labels=axis_labels,
                y_labels=axis_labels,
                matrix_values=self.objective_matrix,
                show_text=True,
            )
        return plots
