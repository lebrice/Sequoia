from typing import (Any, Callable, Generator, Iterable, List, Optional,
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

from settings.active.active_dataloader import ActiveDataLoader
from settings.base.environment import (ActionType, EnvironmentBase,
                                       ObservationType, RewardType)
from utils.logging_utils import get_logger, log_calls

from .gym_dataset import GymDataset
from .utils import ZipDataset

logger = get_logger(__file__)

T = TypeVar("T")
from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter

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

class GymDataLoader(ActiveDataLoader[
        Tensor,
        Tensor,
        Tensor
    ]):
    """ActiveDataLoader made specifically for (possibly batched) Gym envs.
    """
    def __init__(self, env: Union[str, gym.Env],
                       observe_pixels: bool = True,
                       batch_size: int = 1,
                       num_workers: int = 0,
                       worker_init_fn: Optional[Callable[[int], None]] = worker_env_init,
                       collate_fn: Optional[Callable[[List[Tensor]], Tensor]] = collate_fn,
                        **kwargs):
        self.kwargs = kwargs
        # NOTE: This assumes that the 'env' isn't already batched, i.e. that it
        # only returns one observation and one reward per action.
        self.environments: List[GymDataset] = [
            GymDataset(env, observe_pixels=observe_pixels) for _ in range(batch_size)
        ]
        self._observe_pixels = observe_pixels
        self.dataset = ZipDataset(self.environments)
        
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
            # collate_fn=collate_fn,
            **kwargs,
        )

    # def __iter__(self):
    #     if self.num_workers == 0:
    #         return super().__iter__()
    #     else:
    #         return GymMultiprocessingDataLoaderIter(self)

    def __iter__(self):
        for batch in super().__iter__():
            batch = collate_fn(batch)
            # logger.debug(f"batch shape: {batch.shape}")
            yield batch

    def send(self, actions: Tensor) -> Tensor:
        if actions is None or len(actions) == 0:
            # TODO: Should we send random actions to the underlying environments then?
            logger.debug("send method received no action, sending random "
                         "actions to the environments.")
            actions = self.random_actions()
        actions = torch.as_tensor(actions)
        assert len(actions) == len(self.environments), f"# of actions ({len(actions)}) should match # of environments ({len(self.environments)}"
        rewards: List[float] = [
            env.send(action)
            for env, action in zip(self.environments, actions)
        ]
        # logger.debug(f"shapes: {[r.shape for r in rewards]}, dtypes: {[r.dtype for r in rewards]}")
        return torch.as_tensor(rewards)


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
        spaces = [env.observation_space for env in self.environments]
        first_space = spaces[0]
        if not hasattr(first_space, "shape"):
            assert False, first_space
        first_space_shape = first_space.shape
        if not all(getattr(space, "shape", None) == first_space.shape for space in spaces):
            raise RuntimeError(f"Different observation spaces: {spaces}")
        if self.observe_pixels:
            state = self.environments[0].state
            assert state is not None
            return gym.Space(shape=state.shape, dtype=state.dtype)        
        return first_space

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
