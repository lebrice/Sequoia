""" 'Classical' RL setting.
"""
from dataclasses import dataclass
from ..task_incremental_rl_setting import TaskIncrementalRLSetting
from sequoia.utils import constant


@dataclass
class RLSetting(TaskIncrementalRLSetting):
    """ Your usual "Classical" Reinforcement Learning setting.
    
    Implemented as a TaskIncrementalRLSetting, but with a single task.
    """
    nb_tasks: int = constant(1)
