""" Defines the IIDSetting, as a variant of the TaskIncremental setting with
only one task.
"""
from dataclasses import dataclass
from typing import (Callable, ClassVar, Dict, List, Optional, Tuple, Type,
                    TypeVar, Union)

from torch import Tensor
from common.loss import Loss
from common.metrics import Metrics
from common.config import Config
from settings.base import Results
from utils.utils import constant

from .. import TaskIncrementalSetting
from .iid_results import IIDResults

# TODO: Remove the task labels here.

@dataclass
class IIDSetting(TaskIncrementalSetting):
    """Your 'usual' learning Setting, where the samples are i.i.d.
    
    Implemented as a variant of Task-Incremental CL, but with only one task.
    
    """
    results_class: ClassVar[Type[Results]] = IIDResults

    # Held constant, since this is an IID setting.
    nb_tasks: int = constant(1)
    increment: Union[int, List[int]] = constant(None)
    # A different task size applied only for the first task.
    # Desactivated if `increment` is a list.
    initial_increment: int = constant(None)
    # An optional custom class order, used for NC.
    class_order: Optional[List[int]] = constant(None)
    # Either number of classes per task, or a list specifying for
    # every task the amount of new classes (defaults to the value of
    # `increment`).
    test_increment: Optional[Union[List[int], int]] = constant(None)
    # A different task size applied only for the first test task.
    # Desactivated if `test_increment` is a list. Defaults to the
    # value of `initial_increment`.
    test_initial_increment: Optional[int] = constant(None)
    # An optional custom class order for testing, used for NC.
    # Defaults to the value of `class_order`.
    test_class_order: Optional[List[int]] = constant(None)

    def apply(self, method: "Method", config: Config):
        # TODO: Trying to figure out a way to also allow plain/simple
        # pytorch-lightning training in the case of IID settings/methods.
        # if method.target_setting in {TaskIncrementalSetting, ClassIncrementalSetting}:
        #     return super().evaluate(method)
        # else:
        #     # Plain old pytorch-lightning training.
        #     pass
        return super().apply(method, config)

    def test_loop(self, method: "Method") -> IIDResults:
        # TODO: Finish this:
        # TODO: Move split_batch over to the Setting somehow.
        test_metrics = Metrics()
        for x_batch, y_batch in self.test_dataloader():
            predictions = method.get_prediction(x_batch)
            test_metrics += self.get_metrics(y_pred=predictions, y=y_batch)
        return self.results_class(test_metrics=test_metrics)
        
    def train_batch_transforms(self) -> List[Callable]:
        return [
            DropTaskLabels()
        ]

    def valid_batch_transforms(self) -> List[Callable]:
        return self.train_batch_transforms()

    def test_batch_transforms(self) -> List[Callable]:
        return [
            DropTaskLabels()
        ]


@dataclass
class DropTaskLabels(Callable[[Tuple[Tensor, ...]], Tuple[Tensor, ...]]):
    def __call__(self, batch: Tuple[Tensor, ...]):
        if not isinstance(batch, (tuple, list)):
            return batch
        if len(batch) == 3:
            return batch[0], batch[1]
        return batch

SettingType = TypeVar("SettingType", bound=IIDSetting)

if __name__ == "__main__":
    IIDSetting.main()
