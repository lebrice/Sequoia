"""Defines the Abstract Base class for an "Environment".

NOTE (@lebrice): This 'Environment' abstraction isn't super useful at the moment
because there's only the `ActiveDataLoader` that fits this interface (since we
can't send anything to the usual DataLoader).
"""
from abc import ABC, abstractmethod
from typing import (Generic, Iterable, TypeVar)

from utils.logging_utils import get_logger

ObservationType = TypeVar("ObservationType")
ActionType = TypeVar("ActionType")
RewardType = TypeVar("RewardType")

logger = get_logger(__file__)

class EnvironmentBase(Generic[ObservationType, ActionType, RewardType], ABC):
    """ ABC for a learning 'environment', wether RL, Supervised or CL. """
    @abstractmethod
    def __next__(self) -> ObservationType:
        """ Generate the next observation. """

    @abstractmethod
    def __iter__(self) -> Iterable[ObservationType]:
        """ Returns a generator yielding observations and accepting actions. """

    @abstractmethod
    def send(self, action: ActionType) -> RewardType:
        """ Send an action to the environment, and returns the corresponding reward. """

