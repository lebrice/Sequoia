""" Observations/Actions/Rewards particular to an IncrementalSLSetting. 

This is just meant as a cleaner way to import the Observations/Actions/Rewards.
"""
from dataclasses import dataclass
from typing import Optional, TypeVar

from sequoia.settings.assumptions.incremental import IncrementalAssumption
from sequoia.settings.sl.continual.environment import (
    ContinualSLEnvironment as Environment,
)
from sequoia.settings.sl.continual.objects import Observations as ContinualSLObservations
from sequoia.settings.sl.continual.objects import Actions as ContinualSLActions
from sequoia.settings.sl.continual.objects import Rewards as ContinualSLRewards
from sequoia.settings.sl.continual.setting import ContinualSLSetting
from torch import Tensor

# from sequoia.settings.sl.continual.objects import Observations, Actions, Rewards
# from sequoia.settings.assumptions.context_visibility

@dataclass(frozen=True)
class IncrementalSLObservations(ContinualSLObservations):
    """ Incremental Observations, in a supervised context. """
    x: Tensor
    task_labels: Optional[Tensor] = None


@dataclass(frozen=True)
class IncrementalSLActions(ContinualSLActions):
    """Incremental Actions, in a supervised (passive) context."""
    pass


@dataclass(frozen=True)
class IncrementalSLRewards(ContinualSLRewards):
    """Incremental Rewards, in a supervised context."""
    pass


Observations = IncrementalSLObservations
Actions = IncrementalSLActions
Rewards = IncrementalSLRewards
# Environment = C
# Results = IncrementalSLResults

# ObservationType = TypeVar("ObservationType", bound=Observations)
# ActionType = TypeVar("ActionType", bound=Actions)
# RewardType = TypeVar("RewardType", bound=Rewards)

ObservationType = TypeVar("ObservationType", bound=IncrementalSLObservations)
ActionType = TypeVar("ActionType", bound=IncrementalSLActions)
RewardType = TypeVar("RewardType", bound=IncrementalSLRewards)
