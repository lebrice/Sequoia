from typing import *

from torch.utils.data import DataLoader, Dataset, IterableDataset

from sequoia.settings.base.environment import ActionType, Environment, ObservationType, RewardType
from sequoia.utils.logging_utils import get_logger

logger = get_logger(__name__)

from typing_extensions import Final

from .objects import ActionType, ObservationType, RewardType

# TODO: Instead of using a 'y' field for both the supervised learning labels/target and
# for the reward in RL, instead use a 'reward' field in RL, and a 'y' field in SL, where
# in SL the reward could actually be wether the chosen action was correct or not, and
# 'y' could contain the correct prediction for each action.


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
        self.observation: ObservationType = None
        self.action: ActionType = None
        self.reward: RewardType = None

    # def __next__(self) -> ObservationType:
    #     return self.observation

    def send(self, action: ActionType) -> RewardType:
        """Sends an action to the 'dataset'/'Environment'.

        Does nothing when the environment is a simple Dataset (when it isn't an
        instance of EnvironmentBase).

        TODO: Figure out the interactions with num_workers and send, if any.
        """
        self.action = action
        if hasattr(self.dataset, "send"):
            self.reward = self.dataset.send(self.action)
        # TODO: Clean this up, this is taken care of in the GymDataLoader class.
        # if hasattr(self.dataset, "step"):
        #     self.observation, self.reward, self.done, self.info = self.dataset.step(self.action)
        else:
            assert (
                False
            ), "TODO: ActiveDataloader dataset should always have a `send` attribute for now."
        return self.reward


# Deprecated names for the same thing:
ActiveDataLoader = RLEnvironment
ActiveEnvironment = RLEnvironment
