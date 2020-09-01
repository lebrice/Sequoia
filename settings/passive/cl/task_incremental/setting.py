from dataclasses import dataclass
from typing import ClassVar, List, Optional, Type, TypeVar, Union

from torch import Tensor

from common.loss import Loss
from common.transforms import Transforms
from continuum import ClassIncremental, split_train_val
from continuum.datasets import *
from continuum.datasets import _ContinuumDataset
from continuum.scenarios.base import _BaseCLLoader
from settings.base import Results
from settings.passive.cl.setting import (ClassIncrementalSetting,
                                         num_classes_in_dataset)
from settings.passive.environment import PassiveEnvironment
from simple_parsing import list_field
from utils.utils import constant

from .results import TaskIncrementalResults


@dataclass
class TaskIncrementalSetting(ClassIncrementalSetting[Tensor, Tensor]):
    """ Setting where data arrives in a series of Tasks, and where the task
    labels are always available (both train and test time).
    """
    results_class: ClassVar[Type[Results]] = TaskIncrementalResults

    # Wether task labels are available at train time. (Forced to True.)
    task_labels_at_train_time: bool = constant(True)
    # Wether task labels are available at test time.
    # TODO: Is this really always True for all Task-Incremental Settings?
    task_labels_at_test_time: bool = constant(True)

SettingType = TypeVar("SettingType", bound=TaskIncrementalSetting)
