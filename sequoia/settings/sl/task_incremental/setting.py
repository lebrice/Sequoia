""" Defines the Task-Incremental CL Setting.

Task-Incremental CL is a variant of the ClassIncrementalSetting with task labels
available at both train and test time.
"""

from dataclasses import dataclass
from typing import ClassVar, Type, TypeVar

from sequoia.settings.base import Results
from sequoia.utils.utils import constant

from sequoia.settings.sl.incremental import IncrementalSLSetting
from sequoia.settings.sl.incremental import IncrementalSLResults as TaskIncrementalSLResults


@dataclass
class TaskIncrementalSLSetting(IncrementalSLSetting):
    """ Setting where data arrives in a series of Tasks, and where the task
    labels are always available (both train and test time).
    """
    Results: ClassVar[Type[Results]] = TaskIncrementalSLResults

    # Wether task labels are available at train time. (Forced to True.)
    task_labels_at_train_time: bool = constant(True)
    # Wether task labels are available at test time.
    # TODO: Is this really always True for all Task-Incremental Settings?
    task_labels_at_test_time: bool = constant(True)

SettingType = TypeVar("SettingType", bound=TaskIncrementalSLSetting)
