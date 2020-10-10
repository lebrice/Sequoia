"""Defines the Abstract Base class for an "Environment".

NOTE (@lebrice): This 'Environment' abstraction isn't super useful at the moment
because there's only the `ActiveDataLoader` that fits this interface (since we
can't send anything to the usual DataLoader).
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Iterable, Optional, TypeVar

import gym
from torch import Tensor
from torch.utils.data import DataLoader

from common.batch import Batch
from utils.logging_utils import get_logger

logger = get_logger(__file__)


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
    @property
    def prediction(self) -> Tensor:
        return self.y_pred


@dataclass(frozen=True)
class Rewards(Batch):
    """ A batch of "rewards" coming from an Environment.

    For example, in a supervised setting, this would be the true labels, while
    in an RL setting, this would be the 'reward' for a state-action pair.
    """
    y: Optional[Tensor]

    @property
    def labels(self) -> Tensor:
        return self.y

ObservationType = TypeVar("ObservationType", bound=Observations)
ActionType = TypeVar("ActionType", bound=Actions)
RewardType = TypeVar("RewardType", bound=Rewards)


class Environment(gym.Env, Generic[ObservationType, ActionType, RewardType], ABC):
    """ ABC for a learning 'environment', wether RL, Supervised or CL.

    Different settings can implement this interface however they want.
    """
    # TODO: This is currently changing. We don't really need to force the envs
    # on the RL branch to also be iterables/dataloaders, only those on the
    # supervised learning branch. Rather than force RL/active setups to adopt
    # something like __iter__ and send, we can adapt 'active' datasets to be gym
    # Envs instead.
    # The reason why I was considering doing it that way would have been so we
    # could use pytorch lightning Trainers and the other "facilities" meant for
    # supervised learning in RL.
    
    # @abstractmethod
    # def __iter__(self) -> Iterable[ObservationType]:
    #     """ Returns a generator yielding observations and accepting actions. """

    # @abstractmethod
    # def send(self, action: ActionType) -> RewardType:
    #     """ Send an action to the environment, and returns the corresponding reward. """

    # TODO: (@lebrice): Not sure if we really want/need an abstract __next__. 
    # @abstractmethod
    # def __next__(self) -> ObservationType:
    #     """ Generate the next observation. """
