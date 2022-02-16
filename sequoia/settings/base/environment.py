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

logger = get_logger(__name__)

from abc import abstractmethod


class Environment(
    gym.Env,
    Generic[ObservationType, ActionType, RewardType],
    ABC,
):
    """ABC for a learning 'environment' in *both* Supervised and Reinforcement Learning.

    Different settings can implement this interface however they want.
    """

    reward_space: gym.Space

    # @abstractmethod
    def is_closed(self) -> bool:
        """Returns wether this environment is closed."""
        if hasattr(self, "env") and hasattr(self.env, "is_closed"):
            return self.env.is_closed()
        raise NotImplementedError(self)
