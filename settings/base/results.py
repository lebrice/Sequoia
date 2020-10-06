"""In the current setup, `Results` objects are created by a Setting when a
method is applied to them. Each setting can define its own type of `Results` to
customize what the ‘objective’ is in that particular setting.
For instance, the TaskIncrementalSetting class also defines a
TaskIncrementalResults class, where the average accuracy across all tasks is the
objective.

We currently have a unit testing setup that, for a given Method class, performs
a quick run of training / testing (using the --fast_dev_run option from
Pytorch-Lightning).
In those tests, there is also a `validate_results` function, which is basically
used to make sure that the results make sense, for the given method and setting.

For instance, when testing a RandomBaselineMethod on an IIDSetting, the accuracy
should be close to chance level. Likewise, in the `baseline_test.py` file, we
make sure that the BaselineMethod (just a classifier, no CL adjustments) also
exhibits catastrophic forgetting when applied on a Class or Task Incremental
Setting.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import total_ordering
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Type, TypeVar, Union

import matplotlib.pyplot as plt
from simple_parsing import Serializable

from utils.logging_utils import get_logger

logger = get_logger(__file__)


@dataclass
@total_ordering
class Results(Serializable, ABC):
    """ Represents the results of an experiment.
    
    Here you can define what the quantity to maximize/minize is. This class
    should also be used to create the plots that will be helpful to understand
    and compare different results.

    TODO: Add wandb logging here somehow.
    """
    lower_is_better: ClassVar[bool] = False

    @property
    @abstractmethod
    def objective(self) -> float:
        """ Returns a float value that indicating how "good" this result is.
        
        If the `lower_is_better` class variable is set to `False` (default), 
        then this
        """
        raise NotImplementedError("Each Result subclass should implement this.")
    
    @abstractmethod
    def summary(self) -> str:
        """Gives a string describing the results, in a way that is easy to understand.

        :return: A summary of the results.
        :rtype: str
        """

    @abstractmethod
    def make_plots(self) -> Dict[str, plt.Figure]:
        """Generates the plots that are useful for understanding/interpreting or
        comparing this kind of results.

        :return: A dictionary mapping from plot name to the matplotlib figure.
        :rtype: Dict[str, plt.Figure]
        """
    
    # @abstractmethod
    def log_wandb(self) -> Dict[str, Any]:
        """Create a dict version of the results, to be logged to wandb
        """

    def save(self, path: Union[str, Path], dump_fn=None, **kwargs) -> None:
        path = Path(path)
        path.parent.mkdir(exist_ok=True, parents=True)
        return super().save(path, dump_fn=dump_fn, **kwargs)

    def save_to_dir(self,
                    save_dir: Union[str, Path],
                    filename: str = "results.json") -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)

        print(f"Results summary:")
        self.summary
    
        results_dump_file = save_dir / filename
        self.save(results_dump_file)
        print(f"Saved a copy of the results to {results_dump_file}")

        plots: Dict[str, plt.Figure] = self.make_plots()
        plot_paths: Dict[str, Path] = {}
        for fig_name, figure in plots.items():
            print(f"fig_name: {fig_name}")
            # figure.show()
            # plt.waitforbuttonpress(10)
            path = (save_dir/ fig_name).with_suffix(".jpg")
            path.parent.mkdir(exist_ok=True, parents=True)
            figure.savefig(path)
            # print(f"Saved figure at path {path}")
            plot_paths[fig_name] = path
        print(f"\nSaved Plots to: {plot_paths}\n")
            
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Results):
            return self.objective == other.objective
        elif isinstance(other, float):
            return self.objective == other
        return NotImplemented
    
    def __gt__(self, other: Any) -> bool:
        if isinstance(other, Results):
            return self.objective > other.objective
        elif isinstance(other, float):
            return self.objective > other
        return NotImplemented

ResultsType = TypeVar("ResultsType", bound=Results)
