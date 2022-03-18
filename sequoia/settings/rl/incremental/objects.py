from dataclasses import dataclass
from typing import Optional, Sequence, TypeVar, Union

from torch import Tensor

from sequoia.settings.assumptions.incremental import IncrementalAssumption

from ..discrete import DiscreteTaskAgnosticRLSetting

# IncrementalAssumption, DiscreteTaskAgnosticRLSetting


@dataclass(frozen=True)
class Observations(DiscreteTaskAgnosticRLSetting.Observations, IncrementalAssumption.Observations):
    """Observations from a Continual Reinforcement Learning environment."""

    x: Tensor
    task_labels: Optional[Tensor] = None
    # The 'done' that is normally returned by the 'step' method.
    # We add this here in case a method were to iterate on the environments in the
    # dataloader-style so they also have access to those (i.e. for the BaseMethod).
    done: Optional[Union[bool, Sequence[bool]]] = None


@dataclass(frozen=True)
class Actions(DiscreteTaskAgnosticRLSetting.Actions, IncrementalAssumption.Actions):
    """Actions to be sent to a Continual Reinforcement Learning environment."""

    y_pred: Tensor


@dataclass(frozen=True)
class Rewards(DiscreteTaskAgnosticRLSetting.Rewards, IncrementalAssumption.Rewards):
    """Rewards obtained from a Continual Reinforcement Learning environment."""

    y: Tensor


ObservationType = TypeVar("ObservationType", bound=Observations)
ActionType = TypeVar("ActionType", bound=Actions)
RewardType = TypeVar("RewardType", bound=Rewards)
