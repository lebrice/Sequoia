from dataclasses import dataclass
from typing import TypeVar

from torch import Tensor

from sequoia.settings.base import Setting

T = TypeVar("T")


@dataclass(frozen=True)
class Observations(Setting.Observations):
    """Observations in a continual RL Setting."""

    # Input example
    x: Tensor


@dataclass(frozen=True)
class Actions(Setting.Actions):
    pass


# TODO: Replace this 'Rewards' with a 'SparseRewards'-like object for RL, and a
# 'DenseRewards'-like object in SL, rather than use the same in RL and SL.


@dataclass(frozen=True)
class Rewards(Setting.Rewards[T]):
    """Rewards given back by the environment in RL Settings."""


# @dataclass(frozen=True)
# class RLReward(Rewards[T]):
#     reward: T

# @dataclass(frozen=True)
# class SLReward(Rewards[T]):
#     reward: T
#     y: Sequence[T]


ObservationType = TypeVar("ObservationType", bound=Observations)
ActionType = TypeVar("ActionType", bound=Actions)
RewardType = TypeVar("RewardType", bound=Rewards)

# from .environment import RLEnvironment as Environment
