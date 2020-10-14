from abc import abstractmethod
from typing import *

import torch
from torch import multiprocessing as mp
from torch.utils.data import DataLoader, Dataset, IterableDataset

from settings.base.environment import (ActionType, Environment,
                                       ObservationType, RewardType)
from utils.logging_utils import get_logger, log_calls

logger = get_logger(__file__)


class ActiveDataLoader(DataLoader, Environment[ObservationType, ActionType, RewardType]):
    """Extends DataLoader to support sending back actions to the 'dataset'.
    
    TODO: Not really used at the moment besides as a base class for the
    GymDataLoader. 
    
    This could be useful for modeling RL or Active Learning, for instance, where
    the predictions (actions) have an impact on the data generation process.

    When `dataset` isn't an instance of `EnvironmentBase`, i.e. when it is just
    a regular dataset, this doesn't do anything different than DataLoader.

    TODO: Maybe add a custom `map` class for generators?
    
    What's different compared to the usual supervised environment is that
    the observation (x) and the true label (y) are not given at the same time!
    The true label `y` is given only after the prediction is sent back to the dataloader

    TODO: maybe change this class into something like a `FakeActiveEnvironment`.

    """
    def __init__(self, dataset: Union[Dataset, IterableDataset],
                    #    x_transform: Callable = None,
                    #    y_transform: Callable = None,
                       **dataloader_kwargs):
        super().__init__(dataset, **dataloader_kwargs)
        self.observation: ObservationType = None
        self.action: ActionType = None
        self.reward: RewardType = None

        # self.x_transform = x_transform
        # self.y_transform = y_transform

    # def __next__(self) -> ObservationType:
    #     return self.observation

    def send(self, action: ActionType) -> RewardType:
        """ Sends an action to the 'dataset'/'Environment'.
        
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
            assert False, "TODO: ActiveDataloader dataset should always have a `send` attribute for now."
        return self.reward

ActiveEnvironment = ActiveDataLoader
