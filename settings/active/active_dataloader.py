from abc import abstractmethod
from typing import *

import torch
from torch import multiprocessing as mp
from torch.utils.data import DataLoader, Dataset, IterableDataset

from settings.base.environment import (ActionType, EnvironmentBase,
                                       ObservationType, RewardType)
from utils.logging_utils import get_logger, log_calls

logger = get_logger(__file__)


class ActiveDataLoader(DataLoader, EnvironmentBase[ObservationType, ActionType, RewardType]):
    """Extends DataLoader to support sending back actions to the 'dataset'.
    
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
                       x_transform: Callable=None,
                       y_transform: Callable=None,
                       **dataloader_kwargs):
        super().__init__(dataset, **dataloader_kwargs)
        self.observation: ObservationType = None
        self.action: ActionType = None
        self.reward: RewardType = None

        self.x_transform = x_transform
        self.y_transform = y_transform
        self.manager = mp.Manager()
        self.n_pulled: mp.ValueProxy[int] = self.manager.Value(int, 0)
        self.n_pushed: mp.ValueProxy[int] = self.manager.Value(int, 0)

    # @log_calls
    def __next__(self) -> ObservationType:
        # self.observation, self.reward = super().__next__()
        # self.n_pulled.value += 1
        return self.observation

    # def __iter__(self) -> Iterable[ObservationType]:
    #     for batch in super().__iter__():
    #         assert len(batch) == self.batch_size
    #         # The parent dataloader yields both the x's and y's.
    #         self.observation, self.reward = batch
    #         next(self)
    #         y_pred = yield self.observation
    #         if y_pred is not None:
    #             print(f"y_pred: {y_pred}")
    #             y_true = self.send(y_pred)
    
    # @log_calls
    def send(self, action: ActionType) -> RewardType:
        """ Sends an action to the 'dataset'/'Environment'.
        
        Does nothing when the environment is a simple Dataset (when it isn't an
        instance of EnvironmentBase).        
        
        TODO: Figure out the interactions with num_workers and send, if any.
        """
        self.action = action

        # if self.n_pulled.value != (self.n_pushed.value):
        #     raise RuntimeError(
        #         "Number of pulled values should be equal to number of pushed values! "
        #         f"n_pulled: {self.n_pulled.value} n_pushed: {self.n_pushed.value}"
        #     )
        # self.n_pushed.value += 1
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            logger.debug(f"Worker info: {worker_info}")
        else:
            # single-process data loading
            logger.debug("Single process data loading.")

        if isinstance(self.dataset, EnvironmentBase):
            self.reward = self.dataset.send(self.action)
        elif hasattr(self.dataset, "send"):
            self.reward = self.dataset.send(self.action)
        else:
            assert False, "TODO: self.dataset should always be an instance of EnvironmentBase for now."
        return self.reward

ActiveEnvironment = ActiveDataLoader
