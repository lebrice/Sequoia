""" 'Classical' RL setting.
"""
from dataclasses import dataclass
from sequoia.utils import constant

from ..task_incremental_rl_setting import TaskIncrementalRLSetting

@dataclass
class RLSetting(TaskIncrementalRLSetting):
    """ Your usual "Classical" Reinforcement Learning setting.
    
    Implemented as a TaskIncrementalRLSetting, but with a single task.
    """
    nb_tasks: int = constant(1)

    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)
        self._new_random_task_on_reset = True
