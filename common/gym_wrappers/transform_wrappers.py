from typing import Callable, List, Union

import gym
from gym.wrappers import TransformObservation as TransformObservation_
from gym.wrappers import TransformReward as TransformReward_

from utils.logging_utils import get_logger

from .utils import IterableWrapper, reshape_space

logger = get_logger(__file__)

from common.transforms import Compose, Transform


class TransformObservation(TransformObservation_, IterableWrapper):
    def __init__(self, env: gym.Env, f: Union[Callable, Compose]):
        if isinstance(f, list) and not callable(f):
            f = Compose(f)
        super().__init__(env, f=f)
        self.f: Compose
        # Modify the observation space, using the 'space_change' method of Transform.
        self.observation_space = self.env.observation_space
        if isinstance(self.f, Transform) or hasattr(self.f, "space_change"):
            self.space_change = self.f.space_change
            self.observation_space = self.f.space_change(self.observation_space)
            # logger.debug(f"New observation space after transform: {self.observation_space}")
        else:
            logger.warning(UserWarning(f"Don't know how the transform {self.f} will impact the observation space!"))

    
class TransformReward(TransformReward_, IterableWrapper):
    def __init__(self, env: gym.Env, f: Union[Callable, Compose]):
        if isinstance(f, list) and not callable(f):
            f = Compose(f)
        super().__init__(env, f=f)
        self.f: Compose
        # Modify the reward space, if it exists.
        if hasattr(self.env, "reward_space"):
            self.reward_space = self.env.reward_space
            if isinstance(self.f, Transform) or hasattr(self.f, "space_change"):
                self.reward_space = self.f.space_change(self.reward_space)
                logger.debug(f"New reward space after transform: {self.reward_space}")
            else:
                logger.warning(UserWarning(f"Don't know how the transform {self.f} will impact the reward space!"))


class TransformAction(IterableWrapper):
    def __init__(self, env: gym.Env, f: Union[Callable, Compose]):
        if isinstance(f, list) and not callable(f):
            f = Compose(f)
        super().__init__(env)
        self.f: Compose = f
        # Modify the action space, using the 'space_change' method of Transform.
        self.action_space = self.env.action_space
        if isinstance(self.f, Transform) or hasattr(self.f, "space_change"):
            self.action_space = self.f.space_change(self.action_space)
            logger.debug(f"New action space after transform: {self.action_space}")
        else:
            logger.warning(UserWarning(f"Don't know how the transform {self.f} will impact the action space!"))

    def step(self, action):
        return self.env.step(self.action(action))
    
    def action(self, action):
        return self.f(action)
