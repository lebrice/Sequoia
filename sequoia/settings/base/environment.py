"""Defines the Abstract Base class for an "Environment".

NOTE (@lebrice): This 'Environment' abstraction isn't super useful at the moment
because there's only the `ActiveDataLoader` that fits this interface (since we
can't send anything to the usual DataLoader).
"""
from abc import ABC
from typing import Generic

import gym

from sequoia.utils.logging_utils import get_logger

from .objects import ActionType, ObservationType, RewardType

logger = get_logger(__file__)


class Environment(
    gym.Env,
    Generic[ObservationType, ActionType, RewardType],
    ABC,
):
    """ABC for a learning 'environment' in Reinforcement or Supervised Learning.

    Different settings can implement this interface however they want.
    """

    reward_space: gym.Space
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
