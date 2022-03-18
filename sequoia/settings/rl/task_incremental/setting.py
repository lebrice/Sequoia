from dataclasses import dataclass

from sequoia.utils.utils import constant

from ..incremental import IncrementalRLSetting


@dataclass
class TaskIncrementalRLSetting(IncrementalRLSetting):
    """Continual RL setting with clear task boundaries and task labels.

    The task labels are given at both train and test time.
    """

    task_labels_at_train_time: bool = constant(True)
    task_labels_at_test_time: bool = constant(True)
