"""Defines the Abstract Base class for an "Environment".

NOTE (@lebrice): This 'Environment' abstraction isn't super useful at the moment
because there's only the `ActiveDataLoader` that fits this interface (since we
can't send anything to the usual DataLoader).
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (Generic, Iterable, TypeVar, Optional)
from torch import Tensor

from utils.logging_utils import get_logger

logger = get_logger(__file__)

from common.batch import Batch

# TODO: FIXME: We should probably be using something like an Observation / 
# action / reward space! Would that replace or complement these objects?
# Maybe we could actually add a `space` @property on these? Is there such a
# thing as 'optional' dimensions in gym Spaces?


@dataclass(frozen=True)
class Observations(Batch):
    """ A batch of "observations" coming from an Environment. """
    x: Tensor
    @property
    def state(self) -> Tensor:
        return self.x

@dataclass(frozen=True)
class Actions(Batch):
    """ A batch of "actions" coming from an Environment.
    
    For example, in a supervised setting, this would be the predicted labels,
    while in an RL setting, this would be the next 'actions' to take in the
    Environment.
    """
    y_pred: Tensor
    
    @property
    def action(self) -> Tensor:
        return self.y_pred

@dataclass(frozen=True)
class Rewards(Batch):
    """ A batch of "rewards" coming from an Environment.

    For example, in a supervised setting, this would be the true labels, while
    in an RL setting, this would be the 'reward' for a state-action pair.
    """
    y: Optional[Tensor]


ObservationType = TypeVar("ObservationType", bound=Observations)
ActionType = TypeVar("ActionType", bound=Actions)
RewardType = TypeVar("RewardType", bound=Rewards)


class Environment(Generic[ObservationType, ActionType, RewardType], Iterable[ObservationType], ABC):
    """ ABC for a learning 'environment', wether RL, Supervised or CL.

    This defines the basic "interface" of an Environment:
    1. It is an iterable of Observations
    2. You can send actions to it to receive rewards.

    Different settings can implement this interface however they want.
    """
    @abstractmethod
    def __iter__(self) -> Iterable[ObservationType]:
        """ Returns a generator yielding observations and accepting actions. """

    @abstractmethod
    def send(self, action: ActionType) -> RewardType:
        """ Send an action to the environment, and returns the corresponding reward. """

    # TODO: (@lebrice): Not sure if we really want/need an abstract __next__. 
    # @abstractmethod
    # def __next__(self) -> ObservationType:
    #     """ Generate the next observation. """
