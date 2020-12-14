from dataclasses import dataclass
from .class_incremental_rl_setting import ClassIncrementalRLSetting

from sequoia.utils import constant

@dataclass
class TaskIncrementalRLSetting(ClassIncrementalRLSetting):
    """ Continual RL setting with clear task boundaries and task labels.

    The task labels are given at both train and test time.
    """
    task_labels_at_train_time: bool = constant(True)
    task_labels_at_test_time: bool = constant(True)
