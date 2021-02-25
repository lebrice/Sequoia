""" Results of an Incremental setting. """
import json
from io import StringIO
from typing import Dict, List, Union, Optional, ClassVar

import matplotlib.pyplot as plt
import wandb
from sequoia.common.metrics import Metrics

from .iid_results import MetricType, TaskResults


class TaskSequenceResults(List[TaskResults[MetricType]]):
    """ Results for a sequence of Tasks. """
    # For now, all the 'concrete' objectives (mean reward / episode in RL, accuracy in
    # SL) have higher => better 
    lower_is_better: ClassVar[bool] = False
    
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
    """ Results for a whole train loop (transfer matrix).

    This class is basically just a 2d list of TaskResults objects, with some convenience
    methods and properties.
    We get one TaskSequenceResults (a 1d list of TaskResults objects) as a result of
    every test loop, which, in the Incremental Settings, happens after training on each
    task, hence why we get a nb_tasks x nb_tasks matrix of results.
    """
    def __init__(self, *args):
        super().__init__(*args)
        self._runtime: Optional[float] = None

    @property
    def transfer_matrix(self) -> List[List[TaskResults]]:
        return self

    @property
    def metrics_matrix(self) -> List[List[MetricType]]:
        """Returns the 'transfer matrix' but with the average metrics for each task
        in each cell.

        NOTE: This is different from `transfer_matrix` since it returns the matrix of
        `TaskResults` objects (which are themselves lists of Metrics) 

        Returns
        -------
        List[List[MetricType]]
            2d grid of average metrics for each task.
        """
        return [
            [task_results.average_metrics for task_results in task_sequence_result]
            for task_sequence_result in self
        ]

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
        def runtime_score(runtime: float) -> float:
            return 1.0 if runtime <= 3600 else 0.
        score = (
            0.2 * self.average_online_performance.objective +
            0.6 * self.average_final_performance.objective +
            0.2 * runtime_score(self._runtime)
        )
        return score

    @property
    def objective(self) -> float:
        return self.cl_score
    

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
    def online_performance_metrics(self) -> List[MetricType]:
        return [task_results.average_metrics for task_results in self.online_performance]

    @property
    def final_performance(self) -> List[TaskResults[MetricType]]:
        return self.transfer_matrix[-1]

    @property
    def final_performance_metrics(self) -> List[MetricType]:
        return [task_result.average_metrics for task_result in self.final_performance]

    @property
    def average_online_performance(self) -> MetricType:
        return sum(self.online_performance_metrics, Metrics())

    @property
    def average_final_performance(self) -> MetricType:
        return sum(self.final_performance_metrics, Metrics())

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

    def __str__(self) -> str:
        return self.summary()