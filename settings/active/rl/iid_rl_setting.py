""" 'Classical' RL setting.
"""
from dataclasses import dataclass
from .task_incremental_rl_setting import TaskIncrementalRLSetting
from .continual_rl_setting import RemoveTaskLabelsWrapper
from utils import constant


@dataclass
class RLSetting(TaskIncrementalRLSetting):
    """ Your usual "Classical" Reinforcement Learning setting.
    
    Implemented as a TaskIncrementalRLSetting, but with a single task.
    """
    nb_tasks: int = constant(1)

    
    
    def train_dataloader(self, *args, **kwargs):
        env = super().train_dataloader(*args, **kwargs)
        return RemoveTaskLabelsWrapper(env)
    
    def val_dataloader(self, batch_size=None):
        env = super().val_dataloader(batch_size=batch_size)
        return RemoveTaskLabelsWrapper(env)

    def test_dataloader(self, batch_size=None):
        env = super().test_dataloader(batch_size=batch_size)
        return RemoveTaskLabelsWrapper(env)
