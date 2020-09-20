from dataclasses import dataclass
from .task_incremental_rl_setting import TaskIncrementalRLSetting

from utils import constant


@dataclass
class RLSetting(TaskIncrementalRLSetting):
    nb_tasks: int = constant(1)
    
    def train_env_factory(self):
        return self.create_gym_env()
    
    def val_env_factory(self):
        return self.create_gym_env()

    def test_env_factory(self):
        return self.create_gym_env()
