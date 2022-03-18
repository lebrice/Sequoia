from dataclasses import dataclass

from sequoia.utils.utils import constant, flag

from .base import AssumptionBase


@dataclass
class ContinuousContextAssumption(AssumptionBase):
    # Wether we have clear boundaries between tasks, or if the transitions are smooth.
    # Equivalent to wether the context variable is discrete vs continuous.
    smooth_task_boundaries: bool = flag(True)


@dataclass
class DiscreteContextAssumption(ContinuousContextAssumption):
    # Wether we have clear boundaries between tasks, or if the transitions are smooth.
    # Equivalent to wether the context variable is discrete vs continuous.
    smooth_task_boundaries: bool = constant(False)
