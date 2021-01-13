""" IDEA: create the simple train loop for an IID setting (RL or CL).
"""

from .task_incremental import TaskIncrementalSetting
from sequoia.utils import constant
from dataclasses import dataclass


@dataclass
class IIDSetting(TaskIncrementalSetting):
    """ Assumption (mixin) for Settings where the data is stationary (only one
    task).
    """
    nb_tasks: int = constant(1)