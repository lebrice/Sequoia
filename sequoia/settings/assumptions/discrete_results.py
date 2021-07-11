import json
import warnings
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import ClassVar, Dict, Generic, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import wandb
from gym.utils import colorize
from sequoia.common.metrics import Metrics
from sequoia.settings.base.results import Results
from simple_parsing.helpers import list_field

from .iid_results import MetricType, TaskResults


@dataclass
class TaskSequenceResults(Results, Generic[MetricType]):
    """ Results obtained when evaluated on a sequence of (discrete) Tasks. """

    task_results: List[TaskResults[MetricType]] = list_field()

    # For now, all the 'concrete' objectives (mean reward / episode in RL, accuracy in
    # SL) have higher => better
    lower_is_better: ClassVar[bool] = False

    def __post_init__(self):
        if self.task_results and isinstance(self.task_results[0], dict):
            self.task_results = [
                TaskResults.from_dict(task_result, drop_extra_fields=False)
                for task_result in self.task_results
            ]

    @property
    def objective_name(self) -> str:
        return self.average_metrics.objective_name
    
    @property
    def num_tasks(self) -> int:
        """Returns the number of tasks.

        Returns
        -------
        int
            Number of tasks.
        """
        return len(self.task_results)

    @property
    def average_metrics(self) -> MetricType:
        return sum(self.average_metrics_per_task, Metrics())

    @property
    def average_metrics_per_task(self) -> List[MetricType]:
        return [task_result.average_metrics for task_result in self.task_results]

    @property
    def objective(self) -> float:
        return self.average_metrics.objective

    def to_log_dict(self, verbose: bool = False) -> Dict:
        result = {}
        for task_id, task_results in enumerate(self.task_results):
            result[f"Task {task_id}"] = task_results.to_log_dict(verbose=verbose)
        result["Average"] = self.average_metrics.to_log_dict(verbose=verbose)
        return result

    def summary(self, verbose: bool = False):
        s = StringIO()
        print(json.dumps(self.to_log_dict(verbose=verbose), indent="\t"), file=s)
        s.seek(0)
        return s.read()

    def make_plots(self) -> Dict[str, plt.Figure]:
        result = {}
        for task_id, task_results in enumerate(self.task_results):
            result[f"Task {task_id}"] = task_results.make_plots()
        return result

