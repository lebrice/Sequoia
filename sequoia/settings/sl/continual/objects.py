from dataclasses import dataclass
from sequoia.settings.sl import SLSetting
from torch import Tensor
from typing import TypeVar
from sequoia.settings.assumptions.continual import ContinualAssumption


@dataclass(frozen=True)
class Observations(SLSetting.Observations, ContinualAssumption.Observations):
    """ Observations from a Continual Supervised Learning environment. """
    x: Tensor


# TODO: Figure out how to change this perhaps?
# from sequoia.settings.assumptions.task_type import ClassificationActions


@dataclass(frozen=True)
class Actions(SLSetting.Actions):
    """ Actions to be sent to a Continual Supervised Learning environment. """
    y_pred: Tensor

@dataclass(frozen=True)
class Rewards(SLSetting.Rewards):
    """ Rewards obtained from a Continual Supervised Learning environment. """
    y: Tensor


ObservationType = TypeVar("ObservationType", bound=Observations)
ActionType = TypeVar("ActionType", bound=Actions)
RewardType = TypeVar("RewardType", bound=Rewards)
