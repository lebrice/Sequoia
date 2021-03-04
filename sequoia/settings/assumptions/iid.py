""" IDEA: create the simple train loop for an IID setting (RL or CL).
"""

from .task_incremental import TaskIncrementalSetting
from sequoia.utils import constant
from dataclasses import dataclass

# TODO: Import and use the `TaskResults` here.


@dataclass
class IIDSetting(TaskIncrementalSetting):
    """ Assumption (mixin) for Settings where the data is stationary (only one
    task).
    """
    nb_tasks: int = constant(1)
    
    @property
    def phases(self) -> int:
        """The number of training 'phases', i.e. how many times `method.fit` will be
        called.
        
        Defaults to the number of tasks, but may be different, for instance in so-called
        Multi-Task Settings, this is set to 1.
        """
        return 1
