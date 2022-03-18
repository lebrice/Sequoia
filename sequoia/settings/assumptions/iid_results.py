""" Results for an IID experiment. """
from dataclasses import dataclass, field
from typing import ClassVar, Dict, Generic, List, TypeVar

import matplotlib.pyplot as plt

from sequoia.common.metrics import Metrics
from sequoia.settings.base.results import Results

MetricType = TypeVar("MetricType", bound=Metrics)


@dataclass
class TaskResults(Results, Generic[MetricType]):
    """Results within a given Task.

    This is just a List of a given Metrics type, with additional methods.
    """

    # For now, all the 'concrete' objectives (mean reward / episode in RL, accuracy in
    # SL) have higher => better
    lower_is_better: ClassVar[bool] = False

    metrics: List[MetricType] = field(default_factory=list)
    plots_dict: Dict[str, plt.Figure] = field(default_factory=dict)

    def __post_init__(self):
        if self.metrics and isinstance(self.metrics[0], dict):
            self.metrics = [
                Metrics.from_dict(metrics, drop_extra_fields=False) for metrics in self.metrics
            ]

    def __str__(self) -> str:
        return f"{type(self).__name__}(average(metrics)={self.average_metrics})"

    def __repr__(self) -> str:
        return f"{type(self).__name__}(average(metrics)={self.average_metrics})"

    @property
    def average_metrics(self) -> MetricType:
        """Returns the average 'Metrics' object for this task."""
        return sum(self.metrics, Metrics())

    @property
    def objective(self) -> float:
        """Returns the main 'objective' value (a float) for this task.

        This value could be the average accuracy in SL, or the mean reward / episode in
        RL, depending on the type of Metrics stored in `self`.

        Returns
        -------
        float
            A single float that describes how 'good' these results are.
        """
        return self.average_metrics.objective

    @property
    def objective_name(self) -> str:
        # TODO: Add this objective_name attribute on Metrics
        return self.average_metrics.objective_name

    def __str__(self):
        return f"{type(self).__name__}({self.average_metrics})"

    def to_log_dict(self, verbose: bool = False) -> Dict:
        """Produce a dictionary that describes the results / metrics etc.

        Can be logged to console or to wandb using `wandb.log(results.to_log_dict())`.

        Parameters
        ----------
        verbose : bool, optional
            Wether to include very detailed information. Defaults to `False`.

        Returns
        -------
        Dict
            A dict mapping from str keys to either values or nested dicts of the same
            form.
        """
        return self.average_metrics.to_log_dict(verbose=verbose)

    def summary(self) -> str:
        return str(self.to_log_dict())

    def make_plots(self) -> Dict[str, plt.Figure]:
        """Produce a set of plots using the Metrics stored in this object.

        Returns
        -------
        Dict[str, plt.Figure]
            Dict mapping from strings to matplotlib plots.
        """
        # Could actually create plots here too.
        return self.plots_dict
