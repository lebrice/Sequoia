
from .incremental import IncrementalAssumption
from sequoia.utils import constant
from dataclasses import dataclass
from .context_visibility import FullyObservableContextAssumption


@dataclass
class TaskIncrementalAssumption(FullyObservableContextAssumption, IncrementalAssumption):
    """ Assumption (mixin) for Settings where the task labels are available at
    both train and test time.
    """
    task_labels_at_train_time: bool = constant(True)
    task_labels_at_test_time: bool = constant(True)
