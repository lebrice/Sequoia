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
# bob: IterableDataset[ObservationType] = IterableDataset()
from torch.utils.data import DataLoader, IterableDataset

from ..utils.logging_utils import get_logger, log

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


class GymEnvironment(Wrapper, EnvironmentBase[ObservationType, ActionType, RewardType], IterableDataset):
    """ Wrapper around a GymEnvironment that exposes the EnvironmentBase "API"
        and which can be iterated on using DataLoaders.
    """
    def __init__(self, env: gym.Env, observe_pixels: bool=False):
        super().__init__(env=env)
        self.observe_pixels = observe_pixels

        self.action: ActionType
        self.next_state: ObservationType
        self.reward: RewardType
        self.done: bool = False

        self.reset()

        obs = self.env.render(mode="rgb_array")
        self.manager = mp.Manager()
        # Number of steps performed in the environment.
        self._i: mp.Value[int] = self.manager.Value(int, 0)
        self._n_sends: mp.Value[int] = self.manager.Value(int, 0)
        self.action = self.env.action_space.sample()
    
    @log
    def __next__(self) -> ObservationType:
        """ Generate the next observation. """
        self._step(self.action)
        return self.next_state

    @log
    def _step(self, action: ActionType):
        self._i.value += 1

        next_state, self.reward, self.done, self.info = self.env.step(action)
        if self.observe_pixels:
            self.next_state = self.env.render(mode="rgb_array")
        else:
            self.next_state = next_state

    
    @log
    def __iter__(self) -> Generator[ObservationType, ActionType, None]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            logger.debug(f"Worker info: {worker_info}")
        else:
            logger.debug(f"Single process data loading!")
            # single-process data loading:

        while not self.done:
            action = yield next(self)
            if action is not None:
                logger.debug("Received non-None action when yielding?")
                self.action = action
            self._i.value += self.action or 0
    
    @log
    def reset(self):
        start_state = self.env.reset()
        if not self.observe_pixels:
            self.next_state = start_state
        else:
            self.next_state = self.env.render(mode="rgb_array")
        self.action = self.env.action_space.sample()
        self.reward = None

    @log
    def send(self, action: ActionType=None) -> RewardType:
        
        logger.debug(f"Action received at step {self._i}, n_sends = {self._n_sends}: {action}")
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            logger.debug(f"Worker info: {worker_info}")
        else:
            # single-process data loading
            logger.debug("Single process data loading.")
        
        self._n_sends.value += 1
        
        if action is not None:
            self.action = action
        return self.reward


class ZipEnvironments(EnvironmentBase[List[ObservationType], List[ActionType], List[RewardType]], IterableDataset):
    """TODO: Trying to create a 'batched' version of an Environment by creating copies
    of each environment.
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


class ActiveDataLoader(DataLoader):
    def __init__(self,  dataset: EnvironmentBase[ObservationType, ActionType, RewardType], *args, **kwargs):
        super().__init__(dataset, *args, **kwargs)
        assert isinstance(dataset, EnvironmentBase), "dataset should be an instance of EnvironmentBase."
        self.dataset: EnvironmentBase[ObservationType, ActionType, RewardType]

    def send(self, action: ActionType) -> RewardType:
        logger.debug("Send of better dataloader.")
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            logger.debug(f"Worker info: {worker_info}")
        else:
            # single-process data loading
            logger.debug("Single process data loading.")
        return self.dataset.send(action)


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


    @log
    def prepare_data(self, *args, **kwargs):
        super().prepare_data(*args, **kwargs)
    
    @log
    def train_dataloader(self, batch_size: int=None, num_workers: int=0) -> ActiveDataLoader:
        if batch_size not in {None, 1}:
            raise NotImplementedError("Batch size can only be 1 or none for now.")
        batch_size = None
        return ActiveDataLoader(self.env,
            batch_size=None,
            num_workers=num_workers,
            worker_init_fn=self.worker_env_init,
        )

    @log
    def val_dataloader(self, batch_size: int, **kwargs) -> DataLoader:
        return DataLoader(self.env,
            batch_size=batch_size,
            num_workers=0,
            worker_init_fn=self.worker_env_init
        )

    @log
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



# class SupervisedEnvironment(EnvironmentBase):
#     def __init__(self, data_module: LightningDataModule):
#         self.data_module = data_module

# data_dir = Path("data")
# # data_module = MNISTDataModule(data_dir, val_split=5000, num_workers=16, normalize=False)
# # env = SupervisedEnvironment(data_module=data_module)
