import copy
import inspect
import itertools
import logging
import multiprocessing
import os
import textwrap
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial, wraps
from pathlib import Path
from typing import (Any, Callable, ClassVar, Dict, Generator, Generic,
                    Iterable, Iterator, List, NamedTuple, Optional, Tuple,
                    Type, TypeVar, Union)

import gym
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.multiprocessing as mp
from gym import Wrapper
from gym.envs.classic_control import CartPoleEnv
from gym.spaces import Discrete
from pl_bolts.datamodules import LightningDataModule, MNISTDataModule
from pl_bolts.models.rl import DQN
from pl_bolts.models.rl.common.wrappers import make_env
from pytorch_lightning import Trainer, seed_everything
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, IterableDataset

from utils.logging_utils import get_logger, log_calls

ObservationType = TypeVar("ObservationType")
ActionType = TypeVar("ActionType")
RewardType = TypeVar("RewardType")

logger = get_logger(__file__, level=logging.DEBUG)


class EnvironmentBase(ABC, Generator, Generic[ObservationType, ActionType, RewardType]):
    """ ABC for a learning 'environment', wether RL, Supervised or CL. """
    @abstractmethod
    def __next__(self) -> ObservationType:
        """ Generate the next observation. """

    @abstractmethod
    def __iter__(self) -> Generator[ObservationType, ActionType, None]:
        """ Returns a generator yielding observations and accepting actions. """

    @abstractmethod
    def send(self, action: ActionType) -> RewardType:
        """ Send an action to the environment, and returns the corresponding reward. """


class PassiveEnvironment(DataLoader, EnvironmentBase, Generic[ObservationType, RewardType]):
    """Environment in which actions have no influence on future observations.
    
    Normal supervised datasets such as Mnist, ImageNet, etc. fit under this
    category. 
    For now, this is exactly the same as a DataLoader, basically.

    TODO: Could instead subclass the ActiveEnvironment class and add a little
    'mechanism' to yield tuples instead of observations and rewards separately.
    """
    def __next__(self) -> Tuple[ObservationType, RewardType]:
        """ Generate the next observation. """
        return super().__next__()
    
    def __iter__(self) -> Iterable[Tuple[ObservationType, RewardType]]:
        """ Iterate over the environment, yielding batches of Observations (x) and rewards (y) """
        yield from super().__iter__()
    
    def send(self, action: Any) -> None:
        """ Unused, since the environment is passive."""
        pass


class ActiveEnvironment(DataLoader, EnvironmentBase[ObservationType, ActionType, RewardType]):
    """Extends DataLoader to support sending back actions to the 'dataset'.
    
    This could be useful for modeling RL or Active Learning, for instance, where
    the predictions (actions) have an impact on the data generation process.

    When `dataset` isn't an instance of `EnvironmentBase`, i.e. when it is just
    a regular dataset, this doesn't do anything different than DataLoader.

    What's different compared to the usual supervised environment is that
    the observation (x) and the true label (y) are not given at the same time!
    The true label `y` is given only after the prediction is sent back to the
    """
    def __init__(self, dataset: Union[Dataset, IterableDataset], **dataloader_kwargs):
        super().__init__(dataset, **dataloader_kwargs)
        self.observation: Tensor
        self.action: Tensor
        self.reward: Tensor

        self.manager = mp.Manager()
        self.n_pulled: mp.Value[int] = self.manager.Value(int, 0)
        self.n_pushed: mp.Value[int] = self.manager.Value(int, 0)

    @log_calls
    def __next__(self) -> ObservationType:
        self.observation, self.reward = super().__next__()
        self.n_pulled.value += 1
        return self.observation

    @log_calls
    def __iter__(self) -> Generator[ObservationType, ActionType, RewardType]:
        for batch in super().__iter__():
            assert len(batch) == 2, "dataloader should yield both observations and rewards (for now)."
            # The parent dataloader yields both the x's and y's.
            self.observation, self.reward = batch


            x = next(self)
            # Yield x, receive y_pred and give y_true as a 'Reward'.
            y_pred = yield x
            y_true = self.send(y_pred)
    
    @log_calls
    def send(self, action: ActionType) -> RewardType:
        """ Sends an action to the 'dataset'/'Environment'.
        
        Does nothing when the environment is a simple Dataset (when it isn't an
        instance of EnvironmentBase).        
        
        TODO: Figure out the interactions with num_workers and send, if any.
        """
        assert action is not None, "Action shouldn't be None (for now)."
        self.action = action

        if self.n_pulled.value != (self.n_pushed.value + 1):
            raise RuntimeError(
                "Number of pulled values should be equal to number of pushed values + 1! "
                f"n_pulled: {self.n_pulled.value} n_pushed: {self.n_pushed.value}"
            )
        self.n_pushed.value += 1

        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            logger.debug(f"Worker info: {worker_info}")
        else:
            # single-process data loading
            logger.debug("Single process data loading.")

        self.reward = self.dataset.send(action)
        return self.reward


class ZipEnvironments(EnvironmentBase[List[ObservationType], List[ActionType], List[RewardType]], IterableDataset):
    """TODO: Trying to create a 'batched' version of a Generator.
    """
    def __init__(self, *generators: EnvironmentBase[ObservationType, ActionType, RewardType]):
        self.generators = generators
    
    def __next__(self) -> List[ObservationType]:
        return list(next(gen) for gen in self.generators)
    
    def __iter__(self) -> Generator[List[ObservationType], List[ActionType], None]:
        iterators = (
            iter(g) for g in self.generators
        )
        while True:
            actions = yield next(self)

        values = yield from zip(*iterators)
    
    def send(self, actions: List[ActionType]) -> List[RewardType]:
        if action is not None:
            assert len(actions) == len(self.generators)
            self.action = action
        return [
            gen.send(action) for gen, action in zip(self.generators, actions)
        ]


# data_dir = Path("data")
# # data_module = MNISTDataModule(data_dir, val_split=5000, num_workers=16, normalize=False)
# # env = SupervisedEnvironment(data_module=data_module)

class EnvironmentDataModule(LightningDataModule):
    """ Expose a Gym Environment as a LightningDataModule. """

    def __init__(
            self,
            env: EnvironmentBase[ObservationType, ActionType, RewardType],
            train_transforms=None,
            val_transforms=None,
            test_transforms=None,
    ):
        super().__init__(
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
        )
        self.envs: List[GymEnvironment] = []
        self.env = env


    @log_calls
    def prepare_data(self, *args, **kwargs):
        super().prepare_data(*args, **kwargs)
    
    @log_calls
    def train_dataloader(self, batch_size: int=None, num_workers: int=0) -> ActiveEnvironment:
        if batch_size not in {None, 1}:
            raise NotImplementedError("Batch size can only be 1 or none for now.")
        batch_size = None
        return ActiveEnvironment(self.env,
            batch_size=None,
            num_workers=num_workers,
            worker_init_fn=self.worker_env_init,
        )

    @log_calls
    def val_dataloader(self, batch_size: int, **kwargs) -> DataLoader:
        return DataLoader(self.env,
            batch_size=batch_size,
            num_workers=0,
            worker_init_fn=self.worker_env_init
        )

    @log_calls
    def test_dataloader(self, batch_size: int, **kwargs) -> DataLoader:
        return DataLoader(self.env,
            batch_size=batch_size,
            num_workers=0,
            worker_init_fn=self.worker_env_init,
        )

    def worker_env_init(self, worker_id: int):
        logger.debug(f"Initializing dataloader worker {worker_id}")
        worker_info = torch.utils.data.get_worker_info()
        dataset: GymEnvironment = worker_info.dataset  # the dataset copy in this worker process
        
        
        seed = worker_info.seed
        # Sometimes the numpy seed is too large.
        if seed > 4294967295:
            seed %= 4294967295
        logger.debug(f"Seed for worker {worker_id}: {seed}")

        seed_everything(seed)
        
        # TODO: Use this maybe to add an Environemnt in the Batched version of the Environment above?
        # assert len(dataset.envs) == worker_id
        # logger.debug(f"Creating environment copy for worker {worker_id}.")
        # dataset.envs.append(dataset.env_factory())

        # overall_start = dataset.start
        # overall_end = dataset.end
        # configure the dataset to only process the split workload
        # dataset.env_name = ['SpaceInvaders-v0', 'Pong-v0'][worker_info.id]
        # logger.debug(f" ENV: {dataset.env}")
        logger.debug('dataset: ', dataset)

