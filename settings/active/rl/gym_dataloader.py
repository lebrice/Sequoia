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
from .utils import ZipDataset

logger = get_logger(__file__)

T = TypeVar("T")

class GymMultiprocessingDataLoaderIter(_MultiProcessingDataLoaderIter):
    pass



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


def collate_fn(batch: List[Tensor]) -> Tensor:
    assert isinstance(batch, list)
    assert isinstance(batch[0], (Tensor, np.ndarray))
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
    def __init__(self, env: Union[str, gym.Env],
                       observe_pixels: bool = True,
                       batch_size: int = 1,
                       num_workers: int = 0,
                       worker_init_fn: Optional[Callable[[int], None]] = worker_env_init,
                       collate_fn: Optional[Callable[[List[Tensor]], Tensor]] = collate_fn,
                       name: str = "",
                        **kwargs):
        self.kwargs = kwargs
        # NOTE: This assumes that the 'env' isn't already batched, i.e. that it
        # only returns one observation and one reward per action.
        self.environments: List[GymDataset] = [
            GymDataset(env, observe_pixels=observe_pixels) for _ in range(batch_size)
        ]
        self._observe_pixels = observe_pixels
        self.dataset = ZipDataset(self.environments)
        self.name = name
        self.n_sends: int = 0
        self.n_steps: int = 0

        if num_workers not in {0, batch_size}:
            raise RuntimeError(
                "Number of workers should be 0 or batch_size when using a "
                "GymDataLoader."
            )
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
        observations, rewards, dones, infos = zip(*[
            env.step(action)
            for env, action in zip(self.environments, actions)
        ])
        for env, done in zip(self.environments, dones):
            if done:
                env.reset()
        self.observation = torch.as_tensor(observations)
        self.reward = torch.as_tensor(rewards)
        done = False # TODO: When is this dataloader considered 'done'?
        infos = {}
        self.n_steps += 1
        return self.observation, self.reward, done, infos

    def __iter__(self):
        if self.num_workers == 0:
            # return super().__iter__()
            self.step(self.random_actions())
            i = 0
            while True:
                logger.debug(f"Dataloader {self.name}: step {i}")
                assert self.observation is not None
                # Just yield observations, since we assume people are going to
                # call .send(actions) to get rewards.
                self.reward = None
                self.action = None
                action = yield self.observation
                if action is not None:
                    assert False, f'received an action here! {action}'
                
                if self.reward is None:
                    missing_sends = self.n_sends - self.n_steps
                    if missing_sends > 1:
                        raise RuntimeError(
                            f"Dataloader {self.name}: you should have called "
                            f".send() after having received an observation "
                            f"(yielded {i} times, n_steps={self.n_steps}, n_sends={self.n_sends})"
                        )
                    else:
                        logger.warning(RuntimeWarning(
                            f"Dataloader {self.name}: you should have called "
                            f".send() after having received an observation "
                            f"(yielded {i} times, n_steps={self.n_steps}, n_sends={self.n_sends})"
                        ))
                        # we use a random action (just for debugging purposes)
                        # self.send(self.random_actions())
                i += 1

        else:
            raise NotImplementedError("TODO: Implement a cool mp iterator for gym environments.")
            return GymMultiprocessingDataLoaderIter(self)

    def send(self, actions: Tensor) -> Tensor:
        """ Returns the reward associated with the given action, given the current state. """
        logger.debug(f"Dataloader {self.name}: Received actions {actions}, n_sends={self.n_sends}")
        self.action = torch.as_tensor(actions)
        if len(actions) != len(self.environments):
            raise RuntimeError(
                f"# of actions ({len(actions)}) should match # of environments "
                f"({len(self.environments)}"
            )
        self.n_sends += 1
        self.observation, self.reward, _, _ = self.step(actions)
        return self.reward

    def __len__(self) -> Union[int, type(NotImplemented)]:
        # TODO: What should we do in the case where there is no limit?
        return min(
            filter(None, [env.max_steps for env in self.environments]),
            NotImplemented,
        )

    @property
    def action_space(self) -> gym.Space:
        spaces = [env.action_space for env in self.environments]
        first_space = spaces[0]
        if not all(space.shape == first_space.shape for space in spaces):
            raise RuntimeError(f"Different action spaces: {spaces}")
        return first_space

    def random_actions(self) -> np.ndarray:
        """ Returns a batch of random actions. """
        actions = [
            env.action_space.sample() for env in self.environments
        ]
        return torch.as_tensor(actions)

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

    @property
    def observe_pixels(self) -> bool:
        return self._observe_pixels
    
    @observe_pixels.setter
    def observe_pixels(self, value: bool) -> None:
        self._observe_pixels = value
        for env in self.environments:
            env.observe_pixels = value

    def reset(self) -> None:
        for env in self.environments:
            env.reset()

    def close(self) -> None:
        for env in self.environments:
            env.close()

    
    # TODO: Use this maybe to add an Environemnt in the Batched version of the Environment above?
    # assert len(dataset.envs) == worker_id
    # logger.debug(f"Creating environment copy for worker {worker_id}.")
    # dataset.envs.append(dataset.env_factory())

    # overall_start = dataset.start
    # overall_end = dataset.end
    # configure the dataset to only process the split workload
    # dataset.env_name = ['SpaceInvaders-v0', 'Pong-v0'][worker_info.id]
    # logger.debug(f" ENV: {dataset.env}")
    # logger.debug('dataset: ', dataset)
