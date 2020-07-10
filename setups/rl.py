from typing import Generator

import gym
import torch
import torch.multiprocessing as mp
from torch.utils.data import IterableDataset

from ..utils.logging_utils import get_logger, log_calls
from .environment_base import (ActionType, EnvironmentBase, ObservationType,
                               RewardType)

logger = get_logger(__file__)


class GymEnvironment(gym.Wrapper, EnvironmentBase[ObservationType, ActionType, RewardType], IterableDataset):
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
    
    @log_calls
    def __next__(self) -> ObservationType:
        """ Generate the next observation. """
        self._step(self.action)
        return self.next_state

    @log_calls
    def _step(self, action: ActionType):
        self._i.value += 1

        next_state, self.reward, self.done, self.info = self.env.step(action)
        if self.observe_pixels:
            self.next_state = self.env.render(mode="rgb_array")
        else:
            self.next_state = next_state

    
    @log_calls
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
    
    @log_calls
    def reset(self):
        start_state = self.env.reset()
        if not self.observe_pixels:
            self.next_state = start_state
        else:
            self.next_state = self.env.render(mode="rgb_array")
        self.action = self.env.action_space.sample()
        self.reward = None

    @log_calls
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
