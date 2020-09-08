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

from dataclasses import dataclass
from functools import total_ordering
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Type, TypeVar, Union

from common.loss import Loss
from common.metrics import ClassificationMetrics, Metrics, RegressionMetrics
from methods.models import Model
from simple_parsing import Serializable
from utils.logging_utils import get_logger

logger = get_logger(__file__)

R = TypeVar("R", bound="Results")


@dataclass
@total_ordering
class Results(Serializable):
    """ Represents the results of an experiment.
    
    Here you can define what the quantity to maximize/minize is.
    This could be helpful when doing Hyper-Parameter Optimization.

    TODO: @lebrice: Determine which component is "in charge" of determining what
    the objective is: is it the Method? or the Setting?.
    For instance, in a Task-Incremental experiment, the objective is different
    than in an RL experiment.
    """
    hparams: Model.HParams
    test_loss: Loss
    
    lower_is_better: ClassVar[bool] = False

    @property
    def metric(self) -> Metrics:
        """ Gets the most 'important' Metrics object for this results. """
        return self.test_loss.metric

    @property
    def objective(self) -> float:
        """ Returns a float value that measure how good this result is.
        
        """
        metrics = self.metric
        if isinstance(metrics, ClassificationMetrics):
            return metrics.accuracy
        if isinstance(metrics, RegressionMetrics):
            return metrics.mse
        logger.warning(RuntimeWarning(
            "Not sure what the objective is, returning the loss."
        ))
        return float(self.test_loss.loss)

    def save(self, path: Union[str, Path], dump_fn=None, **kwargs) -> None:
        path = Path(path)
        path.parent.mkdir(exist_ok=True, parents=True)
        return super().save(path, dump_fn=dump_fn, **kwargs)
    
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
