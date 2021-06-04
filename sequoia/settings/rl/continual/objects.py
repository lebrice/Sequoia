from dataclasses import dataclass
from sequoia.settings.rl import RLSetting
from torch import Tensor
from typing import TypeVar
from sequoia.settings.assumptions.continual import ContinualAssumption
from .results import ContinualRLResults as Results


@dataclass(frozen=True)
class Observations(RLSetting.Observations, ContinualAssumption.Observations):
    """ Observations from a Continual Supervised Learning environment. """
    x: Tensor


@dataclass(frozen=True)
class Actions(RLSetting.Actions, ContinualAssumption.Actions):
    """ Actions to be sent to a Continual Supervised Learning environment. """
    y_pred: Tensor


@dataclass(frozen=True)
class Rewards(RLSetting.Rewards, ContinualAssumption.Rewards):
    """ Rewards obtained from a Continual Supervised Learning environment. """
    y: Tensor


ObservationType = TypeVar("ObservationType", bound=Observations)
ActionType = TypeVar("ActionType", bound=Actions)
RewardType = TypeVar("RewardType", bound=Rewards)
