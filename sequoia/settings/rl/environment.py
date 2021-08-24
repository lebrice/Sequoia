from abc import abstractmethod
from typing import *

import torch
from torch import multiprocessing as mp
from torch.utils.data import DataLoader, Dataset, IterableDataset

from sequoia.settings.base.environment import (
    ActionType,
    Environment,
    ObservationType,
    RewardType,
)
from sequoia.utils.logging_utils import get_logger, log_calls

logger = get_logger(__file__)

from typing_extensions import Final
from sequoia.settings.base import Rewards


# TODO: Instead of using a 'y' field for both the supervised learning labels/target and
# for the reward in RL, instead use a 'reward' field in RL, and a 'y' field in SL, where
# in SL the reward could actually be wether the chosen action was correct or not, and
# 'y' could contain the correct prediction for each action.
from dataclasses import dataclass

from .objects import ObservationType, ActionType, RewardType


class RLEnvironment(DataLoader, Environment[ObservationType, ActionType, RewardType]):
    """Environment in an RL Setting.

    Extends DataLoader to support sending back actions to the 'dataset'.

    This could be useful for modeling RL or Active Learning, for instance, where
    the predictions (actions) have an impact on the data generation process.

    TODO: Not really used at the moment besides as the base class for the GymDataLoader.
    TODO: Maybe add a custom `map` class for generators?

    Iterating through an RL Environment is different than when iterating on an SL
    environment:
        - Batches only contain the observations, rather than (observations, rewards)
        - The rewards are given back after an action is sent to the environment using
          `send`.

    TODO: maybe change this class into something like a `FakeActiveEnvironment`.

    """

    actions_influence_future_observations: Final[bool] = True

    def __init__(self, dataset: Union[Dataset, IterableDataset], **dataloader_kwargs):
        super().__init__(dataset, **dataloader_kwargs)

    def send(self, action: ActionType) -> RewardType:
        """Sends an action to the 'dataset'/'Environment'.

        Does nothing when the environment is a simple Dataset (when it isn't an
        instance of EnvironmentBase).

        TODO: Figure out the interactions with num_workers and send, if any.
        """
        if not hasattr(self.dataset, "send"):
            raise NotImplementedError(
                "The dataset of RLEnvironment should always have a `send` attribute "
                "for now."
            )
        return self.dataset.send(action)


# Deprecated names for the same thing:
ActiveDataLoader = RLEnvironment
ActiveEnvironment = RLEnvironment
