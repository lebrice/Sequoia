from typing import (Any, Callable, Generator, Iterable, List, Optional,
                    Sequence, Tuple, TypeVar, Union)

import gym
# import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp
from gym.envs.classic_control import CartPoleEnv
from torch.utils.data import Dataset, IterableDataset

from settings.base.environment import (ActionType, EnvironmentBase,
                                       ObservationType, RewardType)
from utils.logging_utils import get_logger, log_calls

logger = get_logger(__file__)

T = TypeVar("T")


class GymDataset(gym.Wrapper, IterableDataset, EnvironmentBase[ObservationType, ActionType, RewardType]):
    """ Wrapper around a GymDataLoaderironment that exposes the EnvironmentBase "API"
        and which can be iterated on using DataLoaders.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, env: Union[str, gym.Env], observe_pixels: bool=True):
        env = gym.make(env) if isinstance(env, str) else env
        super().__init__(env=env)
        self.observe_pixels = observe_pixels

        self.action: ActionType
        self.state: ObservationType
        self.reward: RewardType
        self.done: bool = False

        self.env.render_mode = "rgb_array"

        self.manager = mp.Manager()
        # Number of steps performed in the environment.
        self._i: mp.Value[int] = self.manager.Value(int, 0)
        self._n_sends: mp.Value[int] = self.manager.Value(int, 0)
        self.action = self.env.action_space.sample()
        self.reset()

    # @log_calls
    def __next__(self) -> ObservationType:
        """ Generate the next observation. """
        self.step(self.action)
        return self.state

    # @log_calls
    def step(self, action: ActionType):
        self._i.value += 1

        state, self.reward, self.done, self.info = self.env.step(action)
        if self.observe_pixels:
            self.state = np.asarray(super().render(mode="rgb_array"))
            print(f"state shape: {self.state.shape}")
        else:
            self.state = state
        return self.state, self.reward, self.done, self.info 

    # @log_calls
    @property
    def observation_space(self) -> gym.Space:
        if self.observe_pixels:
            print(f"State shape: {self.state.shape}")
            return gym.Space(shape=self.state.shape, dtype=np.float)
        else:
            return self.env.observation_space
    
    @observation_space.setter
    def observation_space(self, space) -> None:
        self.env.observation_space = space

    # @log_calls
    def __iter__(self) -> Generator[ObservationType, ActionType, None]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            logger.debug(f"Worker info: {worker_info}")
        else:
            pass
            # logger.debug(f"Single process data loading!")
            # single-process data loading:

        while not self.done:
            action = yield next(self)
            if action is not None:
                logger.debug("Received non-None action when yielding?")
                self.action = action
            self._i.value += self.action or 0

    # @log_calls
    def reset(self, **kwargs):
        start_state = super().reset(**kwargs)
        if not self.observe_pixels:
            self.state = start_state
        else:
            self.state = self.env.render(mode="rgb_array")
        self.action = self.env.action_space.sample()
        self.reward = None

    # @log_calls
    def send(self, action: ActionType=None) -> RewardType:
        # logger.debug(f"Action received at step {self._i}, n_sends = {self._n_sends}: {action}")
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            logger.debug(f"Worker info: {worker_info}")
        else:
            pass
            # single-process data loading
            # logger.debug("Single process data loading.")
        
        self._n_sends.value += 1
        
        if action is not None:
            self.action = action
        else:
            # TODO: Take a random action instead?
            self.action = self.action_space.sample()
        return self.reward
    
    def close(self) -> None:
        # plt.close()
        super().close()
