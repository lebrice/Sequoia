from dataclasses import dataclass
from sequoia.utils import constant, flag
from .base import AssumptionBase


@dataclass
class HiddenContextAssumption(AssumptionBase):
    # Wether the task labels are observable during training.
    task_labels_at_train_time: bool = flag(False)
    # Wether the task labels are observable during testing.
    task_labels_at_test_time: bool = flag(False)
    # Wether we get informed when reaching the boundary between two tasks during
    # training.
    known_task_boundaries_at_train_time: bool = flag(False)
    # Wether we get informed when reaching the boundary between two tasks during
    # testing. 
    known_task_boundaries_at_test_time: bool = flag(False)


@dataclass
class PartiallyObservableContextAssumption(HiddenContextAssumption):
    # Wether the task labels are observable during training.
    task_labels_at_train_time: bool = constant(True)
    # Wether we get informed when reaching the boundary between two tasks during
    # training.
    known_task_boundaries_at_train_time: bool = constant(True)
    known_task_boundaries_at_test_time: bool = flag(True)


@dataclass
class FullyObservableContextAssumption(PartiallyObservableContextAssumption):
    # Wether the task labels are observable during testing.
    task_labels_at_test_time: bool = constant(True)
    # Wether we get informed when reaching the boundary between two tasks during
    # testing. 
    known_task_boundaries_at_test_time: bool = constant(True)
