from dataclasses import dataclass
from sequoia.settings.rl import RLSetting
from torch import Tensor
from typing import TypeVar, Optional, Union, Sequence
from sequoia.settings.assumptions.continual import ContinualAssumption
from .results import ContinualRLResults as Results


@dataclass(frozen=True)
class Observations(RLSetting.Observations, ContinualAssumption.Observations):
    """ Observations from a Continual Reinforcement Learning environment. """
    x: Tensor
    task_labels: Optional[Tensor] = None
    # The 'done' that is normally returned by the 'step' method.
    # We add this here in case a method were to iterate on the environments in the
    # dataloader-style so they also have access to those (i.e. for the BaselineMethod).
    done: Optional[Union[bool, Sequence[bool]]] = None


@dataclass(frozen=True)
class Actions(RLSetting.Actions, ContinualAssumption.Actions):
    """ Actions to be sent to a Continual Reinforcement Learning environment. """
    y_pred: Tensor


@dataclass(frozen=True)
class Rewards(RLSetting.Rewards, ContinualAssumption.Rewards):
    """ Rewards obtained from a Continual Reinforcement Learning environment. """
    y: Tensor


ObservationType = TypeVar("ObservationType", bound=Observations)
ActionType = TypeVar("ActionType", bound=Actions)
RewardType = TypeVar("RewardType", bound=Rewards)
