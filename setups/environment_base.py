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

from ..utils.logging_utils import get_logger, log_calls

ObservationType = TypeVar("ObservationType")
ActionType = TypeVar("ActionType")
RewardType = TypeVar("RewardType")

logger = get_logger(__file__, level=logging.DEBUG)


class EnvironmentBase(ABC, Generic[ObservationType, ActionType, RewardType], IterableDataset):
    """ Base class for a learning environment, wether its RL, Supervised or CL.
    """
    @abstractmethod
    def __next__(self) -> ObservationType:
        """ Generate the next observation. """

    @abstractmethod
    def __iter__(self) -> Generator[ObservationType, ActionType, None]:
        """ Iterate over the environment, yielding batches of Observations. """

    @abstractmethod
    def send(self, action: ActionType) -> RewardType:
        """ Send an action to the environment, and returns the corresponding reward. """


class PassiveEnvironment(DataLoader, EnvironmentBase[Tuple[ObservationType, RewardType], None, None]):
    """Environment in which actions have no influence on future observations.
    
    Normal supervised datasets such as Mnist, ImageNet, etc. fit under this category. 
    """
    def __next__(self) -> Tuple[ObservationType, RewardType]:
        """ Generate the next observation. """
        return super().__next__()

    def __iter__(self):
        """ Iterate over the environment, yielding batches of Observations (x) and rewards (y)"""
        yield from super().__iter__()

    def send(self, action: ActionType) -> None:
        """ Unused, since the environment is passive."""
        pass



class ActiveDataLoader(DataLoader):
    """Extends DataLoader to support sending back actions to the 'environment'.

    This could be useful when in RL or Active Learning, for instance.
    When `dataset` isn't an instance of `EnvironmentBase`, i.e. when it is just
    a regular dataset, this doesn't anything different than DataLoader.
    """
    @log_calls
    def send(self, action: ActionType) -> RewardType:
        """ Sends an action to the 'Environment'.
        
        Does nothing when the environment is a simple Dataset (when it isn't an
        instance of EnvironmentBase).        
        
        TODO: Figure out the interactions with num_workers and send, if any.
        """
        # TODO: Should we instead create an `ActiveEnvironment` class and check for it instead? 
        if not isinstance(self.dataset, EnvironmentBase):
            return

        logger.debug("Receiving an action in `send` of ActiveDataloader.")
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            logger.debug(f"Worker info: {worker_info}")
        else:
            # single-process data loading
            logger.debug("Single process data loading.")
        return self.dataset.send(action)


class ActiveEnvironment(ActiveDataLoader, EnvironmentBase[ObservationType, ActionType, RewardType]):
    """ Environment where actions have an impact on future observations.
    
    This can be used to model an RL environment, for instance.

    What's different compared to the usual supervised environment is that
    the observation (x) and the true label (y) are not given at the same time!
    The true label `y` is given only after the prediction is sent back to the

    For instance:
    ```python
    my_model = MyModel()
    dataset = MNIST("data")
    env = ActiveEnvironment(data_source=dataset)
    for x in env:
        y_pred = my_model(x)
        # 'send' the prediction (_action_), obtain a label (_reward_)
        y_true = env.send(y_pred)
        loss = my_model.get_loss(y_true, y_pred)
        loss.backward()
        my_model.optimizer_step()
    ```
    """
    def __init__(self, data_source: Union[Dataset, ActiveDataLoader], **dataloader_kwargs):
        if isinstance(data_source, Dataset):
            data_source = ActiveDataLoader(data_source, **dataloader_kwargs)
        self.dataloader = data_source

        self.observation: Tensor
        self.action: Tensor
        self.reward: Tensor

        self.dataloader_iter = iter(self.dataloader)
        self.manager = mp.Manager()
        self.n_pulled: mp.Value[int] = self.manager.Value(int, 0)
        self.n_pushed: mp.Value[int] = self.manager.Value(int, 0)

    @log_calls
    def __next__(self) -> int:
        self.x, self.y_true = next(self.dataloader_iter)
        self.n_pulled.value += 1
        return self.x

    @log_calls
    def __iter__(self) -> Generator[Tensor, Tensor, Tensor]:
        y_true_prev: Optional[Tensor] = None
        while True:
            x = next(self)
            # Yield x, receive y_pred and give y_true as a 'Reward'.
            y_pred = yield x, y_true_prev
            y_true = self.send(y_pred)
            
    @log_calls
    def send(self, action: int) -> int:
        self.y_pred = action
        if self.n_pulled.value != (self.n_pushed.value + 1):
            raise RuntimeError(
                "Number of pulled values should be equal to number of pushed values + 1! "
                f"n_pulled: {self.n_pulled.value} n_pushed: {self.n_pushed.value}"
            )
        self.n_pushed.value += 1
        return self.y_true


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
    def train_dataloader(self, batch_size: int=None, num_workers: int=0) -> ActiveDataLoader:
        if batch_size not in {None, 1}:
            raise NotImplementedError("Batch size can only be 1 or none for now.")
        batch_size = None
        return ActiveDataLoader(self.env,
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

