""" Defines the Task-Incremental CL Setting.

Task-Incremental CL is a variant of the ClassIncrementalSetting with task labels
available at both train and test time.
"""

from dataclasses import dataclass
from typing import ClassVar, Type, TypeVar

from settings.base import Results
from utils.utils import constant

from ..class_incremental_setting import ClassIncrementalSetting
from . import TaskIncrementalResults


@dataclass
class TaskIncrementalSetting(ClassIncrementalSetting):
    """ Setting where data arrives in a series of Tasks, and where the task
    labels are always available (both train and test time).
    """
    Results: ClassVar[Type[Results]] = TaskIncrementalResults

    # Wether task labels are available at train time. (Forced to True.)
    task_labels_at_train_time: bool = constant(True)
    # Wether task labels are available at test time.
    # TODO: Is this really always True for all Task-Incremental Settings?
    task_labels_at_test_time: bool = constant(True)

SettingType = TypeVar("SettingType", bound=TaskIncrementalSetting)
