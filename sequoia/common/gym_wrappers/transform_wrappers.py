from typing import Callable, Union

import gym
from gym import Space, spaces
from gym.wrappers import TransformObservation as TransformObservation_
from gym.wrappers import TransformReward as TransformReward_

from sequoia.common.gym_wrappers.convert_tensors import (add_tensor_support,
                                                         has_tensor_support)
from sequoia.common.transforms import Compose, Transform
from sequoia.utils.logging_utils import get_logger

from .utils import IterableWrapper

logger = get_logger(__file__)


class TransformObservation(TransformObservation_, IterableWrapper):
    def __init__(self, env: gym.Env, f: Union[Callable, Compose]):
        if isinstance(f, list) and not callable(f):
            f = Compose(f)
        super().__init__(env, f=f)
        self.f: Transform
        # try:
        self.observation_space = self(self.env.observation_space)
        if has_tensor_support(self.env.observation_space):
            self.observation_space = add_tensor_support(self.observation_space)
        # except Exception as e:
            # logger.warning(UserWarning(
            #     f"Don't know how the transform {self.f} will impact the "
            #     f"observation space! (Exception: {e})"
            # ))

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

    def __iter__(self):
        if self.wrapping_passive_env:
            # TODO: For now, we assume that the passive environment has already
            # split stuff correctly for us to use.
            for obs, rewards in self.env:
                yield self(obs), rewards
        else:
            return super().__iter__()


class TransformReward(TransformReward_, IterableWrapper):
    def __init__(self, env: gym.Env, f: Union[Callable, Compose]):
        if isinstance(f, list) and not callable(f):
            f = Compose(f)
        super().__init__(env, f=f)
        self.f: Compose
        # Modify the reward space, if it exists.
        if hasattr(self.env, "reward_space"):
            self.reward_space = self.env.reward_space
        else:
            self.reward_space = spaces.Box(
                low=self.env.reward_range[0],
                high=self.env.reward_range[1],
                shape=(),
            )

        try:
            self.reward_space = self.f(self.reward_space)
            logger.debug(f"New reward space after transform: {self.reward_space}")
        except Exception as e:
            logger.warning(UserWarning(
                f"Don't know how the transform {self.f} will impact the "
                f"observation space! (Exception: {e})"
            ))


class TransformAction(IterableWrapper):
    def __init__(self, env: gym.Env, f: Callable[[Union[gym.Env, Space]], Union[gym.Env, Space]]):
        if isinstance(f, list) and not callable(f):
            f = Compose(f)
        super().__init__(env)
        self.f: Compose = f
        # Modify the action space by applying the transform onto it.
        self.action_space = self.env.action_space

        if isinstance(self.f, Transform):
            self.action_space = self.f(self.env.action_space)
            # logger.debug(f"New action space after transform: {self.observation_space}")

    def step(self, action):
        return self.env.step(self.action(action))

    def action(self, action):
        return self.f(action)
