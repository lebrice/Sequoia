from typing import (Any, Callable, Dict, Generator, Iterable, List, Optional,
                    Sequence, Tuple, Type, TypeVar, Union)

import gym
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp
from gym.envs.classic_control import CartPoleEnv
from gym.spaces import Box, Discrete
from gym.wrappers.pixel_observation import PixelObservationWrapper
from torch import Tensor
from torch.utils.data import Dataset, IterableDataset

from common.gym_wrappers import PixelStateWrapper, wrapper_is_present
from settings.base.environment import (ActionType, EnvironmentBase,
                                       ObservationType, RewardType)
from utils.logging_utils import get_logger, log_calls

logger = get_logger(__file__)

T = TypeVar("T")



class GymDataset(gym.Wrapper, IterableDataset, EnvironmentBase[ObservationType, ActionType, RewardType]):
    """ Wrapper around a GymDataLoaderironment that exposes the EnvironmentBase "API"
        and which can be iterated on using DataLoaders.
    """
    # metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self,
                 env: Union[str, gym.Env],
                 observe_pixels: bool = False,
                 max_episodes: Optional[int] = None,
                 max_steps: Optional[int] = None,
                 ):
        env = gym.make(env) if isinstance(env, str) else env
        env.reset()

        if observe_pixels and not wrapper_is_present(env, PixelStateWrapper):
            logger.debug(f"Adding a Wrapper to {env} to observe pixels rather than internal state.")
            env = PixelStateWrapper(env)
        super().__init__(env=env)

        self.env: gym.Env
        self.observe_pixels = observe_pixels
        self.action: ActionType = self.env.action_space.sample()
        self.state: ObservationType
        self.reward: RewardType
        self.done: bool = False
        self.info: Dict = {}

        # Number of steps performed in the environment.
        self.step_count: int = 0
        # Number of times the `send` method was called, i.e. number of actions
        # taken in the environment.
        self.send_count: int = 0
        # Maximum number of steps to perform in the environment.
        self.max_steps: Optional[int] = max_steps
        # Number of episodes performed in the environment.
        self.episode_count: int = 0
        # Maximum number of episodes to perform in the environment.
        self.max_episodes: Optional[int] = max_episodes

        self.total_reward: float = 0.
        self.reset()

    def __next__(self) -> ObservationType:
        """ Generate the next observation. """
        self.step(self.action)
        return self.state

    def step(self, action: ActionType):
        # TODO: Should we maybe move some of this to a Wrapper? (Maybe something
        # like this already exists somewhere else also)
        if isinstance(action, Tensor):
            action = action.cpu().numpy()
        if not self.action_space.contains(action):
            if isinstance(self.action_space, Discrete):
                # TODO: If the action is the scores of a classifier, pick the 
                # argmax as the action?
                action = round(float(action))
            elif isinstance(self.action_space, Box):
                action = np.clip(action, self.action_space.low, self.action_space.high)
            else:
                # logger.warning(RuntimeWarning(f"The action space {self.action_space} doesn't contain action {action}!"))
                pass
        self.state, self.reward, self.done, self.info = super().step(action)
        self.step_count += 1
        self.total_reward += self.reward
        return self.state, self.reward, self.done, self.info

    @property
    def reached_step_limit(self) -> bool:
        if self.max_steps:
            return self.step_count >= self.max_steps
        return False
    
    @property
    def reached_episode_limit(self) -> bool:
        if self.max_episodes:
            return self.episodes_count >= self.max_episodes
        return False


    def __iter__(self) -> Generator[ObservationType, ActionType, None]:
        action: ActionType = self.action_space.sample()
        
        while not self.reached_episode_limit or self.reached_step_limit:
            # logger.debug(f"n steps: {self.step_count}, n sends: {self.send_count}")
            # Perform an episode.
            while not (self.done or self.reached_step_limit):
                # Isn't there something fishy going on here? I'm not sure. Are
                # we giving back the right reward for the right action and observation?
                action = yield self.__next__()

                assert self.reward is not None
                if action is not None:
                    assert False, "Received non-None action (from send on the iterator, rather than on `self`!"
                    self.action = action

            self.episode_count += 1
            # NOTE: It's important that we call `self.env.reset` rather than
            # `self.reset` because that would reset the number of episodes and
            # steps performed and mess up the checks above.
            self.env.reset()

    def reset(self, **kwargs) -> ObservationType:
        start_state = super().reset(**kwargs)
        self.state = start_state
        self.action = self.env.action_space.sample()
        # TODO: Should we reset the number of steps? This doesn't sound right.
        # self.step_count = 0
        # self.episode_count = 0
        # self.send_count = 0
        self.reward = None
        return self.state

    def send(self, action: ActionType) -> RewardType:
        # TODO: There might be somethign wrong here. What we're basically doing
        # is giving the 'user' back the reward for the 'previous' action, not
        # for the current one being sent!
        if action is None:
            assert False, f"Don't send a None action!"
            # TODO: Take a random action instead?
            action = self.action_space.sample()
        
        # TODO: Need to check that 'action' (which might be a Tensor) into
        # the kind of thing the underlying gym environment expects!
        # Could probably use one of the gym wrappers for this, actually!
        if not self.action_space.contains(action):
            if isinstance(self.action_space, Discrete):
                action = round(float(action))
            elif isinstance(self.action_space, Box):
                action = np.clip(action, self.action_space.low, self.action_space.high)
            else:
                # logger.warning(RuntimeWarning(f"The action space {self.action_space} doesn't contain action {action}!"))
                pass
        self.action = action
        # TODO: Determine if we shoud call 'self.step' here or in __next__ ??
        # Could do something where we compare n_sends and n_steps?
        # self.state, self.reward, self.done, self.info = self.step(self.action)
        self.send_count += 1
        self.action = action
        return self.reward

    def close(self) -> None:
        # plt.close()
        super().close()
