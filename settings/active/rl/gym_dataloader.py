import textwrap
from typing import (Any, Callable, Dict, Generator, Iterable, List, Optional,
                    Sequence, Tuple, TypeVar, Union)

import gym
# import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp
from gym.envs.classic_control import CartPoleEnv
from pytorch_lightning import seed_everything
from torch import Tensor
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter

from settings.active.active_dataloader import ActiveDataLoader
from settings.base.environment import (ActionType, EnvironmentBase,
                                       ObservationType, RewardType)
from utils.logging_utils import get_logger, log_calls

from .gym_dataset import GymDataset
from .zip_dataset import ZipDataset

logger = get_logger(__file__)

T = TypeVar("T")




def worker_env_init(worker_id: int):
    """ TODO: Experimenting with using a worker_init_fn arg to DataLoader to for
    multiple workers with active (Gym) environments.
    """
    # TODO: Is this needed when using pytorch lightning?
    logger.debug(f"Initializing dataloader worker {worker_id}")
    worker_info = torch.utils.data.get_worker_info()
    # the dataset copy in this worker process
    dataset: GymDataset = worker_info.dataset
    logger.debug(f"dataset type: {type(dataset)}, id: {id(dataset)}")
    seed = worker_info.seed
    # Sometimes the numpy seed is too large.
    if seed > 4294967295:
        seed %= 4294967295
    logger.debug(f"Seed for worker {worker_id}: {seed}")
    seed_everything(seed)

    # TODO: Use this maybe to add an Environemnt in the Batched version of the Environment above?
    # assert len(dataset.datasets) == worker_id
    # logger.debug(f"Creating environment copy for worker {worker_id}.")
    # assert False
    # dataset.datasets.append(dataset.env_factory())

    # overall_start = dataset.start
    # overall_end = dataset.end
    # configure the dataset to only process the split workload
    # dataset.env_name = ['SpaceInvaders-v0', 'Pong-v0'][worker_info.id]
    # logger.debug(f" ENV: {dataset.env}")
    # logger.debug('dataset: ', dataset)


def collate_fn(batch: List[Tensor]) -> Tensor:
    assert isinstance(batch, list)
    assert isinstance(batch[0], (Tensor, np.ndarray))
    logger.debug(f"Inside collate function!")
    # logger.debug(f"observation shapes: {[v.shape for v in batch]}")
    batch = [torch.as_tensor(obs, dtype=float) for obs in batch]
    if batch[0].shape[0] == 1 and batch[0].ndim > 1:
        batch = torch.cat(batch)
    else:
        batch = torch.stack(batch)
    return batch


class GymDataLoader(ActiveDataLoader[Tensor, Tensor, Tensor]):
    """ActiveDataLoader made specifically for (possibly batched) Gym envs.
    """
    def __init__(self,
                 env: str = None,
                 env_factory: Callable[[], gym.Env] = None,
                 observe_pixels: bool = False,
                 transforms: Optional[Callable] = None,
                 batch_size: int = 1,
                 num_workers: int = 0,
                 max_steps: int = 10_000,
                 random_actions_when_missing: bool = True,
                 policy: Callable[[Tensor], Tensor] = None,
                 worker_init_fn: Optional[Callable[[int], None]] = worker_env_init,
                 collate_fn: Optional[Callable[[List[Tensor]], Tensor]] = collate_fn,
                 name: str = "",
                  **kwargs):
        assert env or env_factory, "One of `env` or `env_factory` must be set."
        self.transforms = transforms
        self.kwargs = kwargs
        # NOTE: This assumes that the 'env' isn't already batched, i.e. that it
        # only returns one observation and one reward per action.
        self.environments: List[GymDataset] = [
            GymDataset(
                env if env else env_factory(),
                observe_pixels=observe_pixels
            ) for _ in range(batch_size)
        ]
        self._observe_pixels = observe_pixels
        self.name = name
        # Counts when an action is sent back to the dataloader using send()
        self.n_sends: int = 0
        # Counts when the underlying environments are updated using step()
        self.n_steps: int = 0
        # Number of random actions that were created to 'fill-in' a missing call
        # to .send(). (This is only incremented when
        # `random_actions_when_missing` is True.
        self.n_random: int = 0
        # Maximum total number of steps to perform. When we reach this, we raise
        # a StopIteration in __iter__
        self.max_steps: int = max_steps
        # If True, when trying to pull two batches of observations from the
        # dataloader (one after another, without sending back an action between
        # them), we apply random actions to each underlying environment in order
        # to actions to each underlying environment 
        # to generate the second batch of observations. 
        self.random_actions_when_missing = random_actions_when_missing
        # a Policy to use whenever actions aren't explicitly sent to the
        # dataloader.
        self.policy: Callable[[Tensor], Tensor] = policy
        assert not (self.random_actions_when_missing and self.policy), (
            f"Can only use one of `random_actions_when_missing` and `policy`."
        )
        self.done: bool = False

        if num_workers not in {0, batch_size}:
            raise RuntimeError(
                "Number of workers should be 0 or batch_size when using a "
                "GymDataLoader."
            )
        # This 'dataset' attribute isn't really used atm.
        self.dataset = ZipDataset(self.environments)
        # init the dataloader.
        super().__init__(
            self.dataset,
            # TODO: Debug the multi-worker data-loading with Gym Dataloaders.
            # num_workers=num_workers,
            num_workers=0,
            # Set this to 'None' to signify that we take care of the batching
            # ourselves.
            batch_size=None,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn,
            **kwargs,
        )

    def step(self, actions: Tensor) -> Tuple[Tensor, Tensor, bool, Dict]:
        # TODO: Do this more efficiently, following something like 
        # https://squadrick.dev/journal/efficient-multi-gym-environments.html
        observations, rewards, dones, infos = zip(*[
            env.step(action)
            for env, action in zip(self.environments, actions)
        ])
        # TODO: Meed to double check that the 'reset'-ed state is in line with
        # the actions. For instance, maybe return `dones` as a bool mask so the
        # model can ignore some of the loss terms and not learn something dumb?
        for env, done in zip(self.environments, dones):
            if done:
                env.reset()
        self.observation = torch.as_tensor(observations)
        if self.transforms:
            # Apply the transforms to the observations.
            self.observation = self.transforms(self.observation)

        self.reward = torch.as_tensor(rewards)
        infos = {}
        self.n_steps += 1
        self.done = self.n_steps >= self.max_steps # TODO: When is this dataloader considered 'done'?
        return self.observation, self.reward, done, infos
    
    def __iter__(self):
        if self.num_workers == 0:
            # This would basically just return the entries from `self.dataset`.
            # return super().__iter__()
            return iter(GymSingleProcessDataLoaderIter(self))            
        else:
            # raise NotImplementedError("TODO: Implement a cool mp iterator for gym environments.")
            return GymMultiprocessingDataLoaderIter(self)

    def send(self, actions: Optional[Tensor]) -> Tensor:
        """ Returns the reward associated with the given action, given the current state. """
        # logger.debug(f"Dataloader {self.name}: Received actions {actions}, n_sends={self.n_sends}")
        if actions is None:
            actions = self.random_actions()
            self.n_random += 1
        
        self.action = torch.as_tensor(actions)
        if len(actions) != len(self.environments):
            raise RuntimeError(
                f"# of actions ({len(actions)}) should match # of environments "
                f"({len(self.environments)}"
            )
        self.n_sends += 1
        logger.debug(f"actions: {actions}")
        self.observation, self.reward, _, _ = self.step(actions)
        return self.reward

    def __len__(self) -> int:
        # TODO: What should we do in the case where there is no limit?
        return self.max_steps

    def random_actions(self) -> np.ndarray:
        """ Returns a batch of random actions. """
        return torch.as_tensor([
            env.action_space.sample() for env in self.environments
        ])

    @property
    def observation_space(self) -> gym.Space:
        # Assumes that all the spaces are the same.
        assert self.environments
        return self.environments[0].observation_space

    @property
    def action_space(self) -> gym.Space:
        # Assumes that all the spaces are the same.
        assert self.environments
        return self.environments[0].action_space

    @property
    def batch_size(self) -> Optional[int]:
        return len(self.environments)
    
    @batch_size.setter
    def batch_size(self, value: int) -> None:
        if not self.num_workers and value and len(self.environments) != value:
            raise RuntimeError(
                f"Can't change the batch size (yet). Current batch size: "
                f"{len(self.environments)}, new: {value}"
            )
        assert value is None, (
            f"Can't change the batch size (yet). Current batch size: "
            f"{len(self.environments)}, new: {value}"
        )

    def reset(self, **kwargs) -> List[Any]:
        """ Resets the environments.

        NOTE: This doesn't reset the state of the dataloader itself (n_steps,
        n_sends, etc).
        """
        start_states = torch.as_tensor([
            env.reset(**kwargs)
            for env in self.environments
        ])
        return start_states

    def close(self) -> None:
        for env in self.environments:
            env.close()

    def seed(self, value: int) -> None:
        for env in self.environments:
            env.seed(value)


class GymSingleProcessDataLoaderIter():
    def __init__(self, loader: GymDataLoader):
        self.loader = loader

    def __next__(self):
        return self.loader.observation

    def __iter__(self):
        self.loader.observation = self.loader.reset()
        if self.loader.transforms:
            self.loader.observation = self.loader.transforms(self.loader.observation)

        for i in range(self.loader.max_steps):
            logger.debug(
                f"Dataloader {self.loader.name}: step {i}, "
                f"n_steps={self.loader.n_steps}, n_sends={self.loader.n_sends} "
                f"n_random={self.loader.n_random}, self.loader.observation.shape: {self.loader.observation.shape}"
            )
            if i != self.loader.n_steps:
                if self.loader.random_actions_when_missing:
                    # TODO: This doesn't work, because we can't give back
                    # the reward this way!
                    reward = self.loader.send(self.loader.random_actions())
                elif self.loader.policy is not None:
                    action = self.loader.policy(self.loader.observation)
                    reward = self.loader.send(action)
                else:
                    raise RuntimeError(
                        f"Dataloader {self.loader.name}: you need to send a batch "
                        f"of actions back to the GymDataLoader each time "
                        f"you get an observation, using `.send(actions)` "
                        f"method. Alternatively, consider setting "
                        f"`random_actions_when_missing` to True.\n"
                        f"(yielded {i} observations, received "
                        f"{self.loader.n_sends} actions)"
                    )

            # Set those to 'None', to force users to call the either the
            # `send(actions)` or `step(action)` method (preferably `send`!) to get a
            # reward for a given action.
            self.loader.reward = None
            self.loader.action = None
            # Only yield observations, since we assume users are going to
            # call .send(actions) to get the associated reward.
            action = yield self.loader.observation

            # If we received something from the yield statement, then the
            # user is interacting with the iterator as a generator! We can't
            # have that. We want users to call `.send(action)` on the actual
            # DataLoader, not on the iterator object! (because if that were
            # the case, we'd have to yield both the observations and the
            # rewards!)  
            if action is not None:
                assert False, f'received an action here! {action}'
            
            if self.loader.reward is None:
                missing_sends = self.loader.n_sends - self.loader.n_steps
                logger.debug(f"Missing sends: {missing_sends}.")
            i += 1


class GymMultiprocessingDataLoaderIter(_MultiProcessingDataLoaderIter):
    pass