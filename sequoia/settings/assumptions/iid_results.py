""" Results for an IID experiment. """
from typing import TypeVar, List, Dict
import matplotlib.pyplot as plt
from sequoia.common.metrics import Metrics


MetricType = TypeVar("MetricType", bound=Metrics)


class TaskResults(List[MetricType]):
    """ Results within a given Task.

    This is just a List of a given Metrics type, with additional methods.
    """

    @property
    def average_metrics(self) -> MetricType:
        """ Returns the average 'Metrics' object for this task. """
        return sum(self, Metrics())

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

    def make_plots(self) -> Dict[str, plt.Figure]:
        """Produce a set of plots using the Metrics stored in this object.

        Returns
        -------
        Dict[str, plt.Figure]
            Dict mapping from strings to matplotlib plots.
        """
        return {}
