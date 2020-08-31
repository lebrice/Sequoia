import itertools
from dataclasses import dataclass
from typing import Dict, List

from common.loss import Loss
from common.metrics import ClassificationMetrics, Metrics, RegressionMetrics
from utils.logging_utils import get_logger
from simple_parsing import list_field
from .. import Results

logger = get_logger(__file__)

@dataclass
class ClassIncrementalResults(Results):
    task_losses: List[Loss] = list_field()
    @property
    def task_metrics(self) -> List[Metrics]:
        """ Gets the Metrics for each task. """
        # print(f"Loss names: {[loss.name for loss in self.task_losses]}")
        # for loss in self.task_losses:
        #     print(loss.name)
        #     print(loss.losses.keys())
        #     for name, metric in loss.all_metrics().items():
        #         print(name, metric)
        return [loss.metric for loss in self.task_losses]

    @property
    def metric(self) -> Metrics:
        """ Gets the most 'important' Metrics object for this results. """
        return sum(self.task_metrics, Metrics())

    @property
    def objective(self) -> float:
        metrics = self.metric
        if isinstance(metrics, ClassificationMetrics):
            return metrics.accuracy
        elif isinstance(metrics, RegressionMetrics):
            return metrics.mse
        else:
            logger.error(
                f"Not sure what the objective is, returning the loss. "
                f"(self.hparams={self.hparams}, self.test_loss={self.test_loss})")
            return float(self.test_loss.loss)
