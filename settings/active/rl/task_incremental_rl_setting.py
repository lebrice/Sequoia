from dataclasses import dataclass
from .class_incremental_rl_setting import ClassIncrementalRLSetting

from utils import constant

@dataclass
class TaskIncrementalRLSetting(ClassIncrementalRLSetting):
    task_labels_at_train_time: bool = constant(True)
    task_labels_at_test_time: bool = constant(True)
