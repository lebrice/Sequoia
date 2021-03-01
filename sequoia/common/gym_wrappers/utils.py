from abc import ABC
from typing import (Dict, Generic, Iterator, List, NamedTuple, Tuple, Type,
                    Union, TypeVar)
from collections.abc import Sized
from functools import singledispatch
import gym
import numpy as np
from gym import spaces
from gym.vector.utils import batch_space
from torch.utils.data import IterableDataset, DataLoader

from sequoia.utils.logging_utils import get_logger


from gym.envs.classic_control import AcrobotEnv, CartPoleEnv, PendulumEnv, MountainCarEnv, Continuous_MountainCarEnv
classic_control_envs = (AcrobotEnv, CartPoleEnv, PendulumEnv, MountainCarEnv, Continuous_MountainCarEnv)

classic_control_env_prefixes: Tuple[str, ...] = (
    "CartPole", "Pendulum", "Acrobot", "MountainCar", "MountainCarContinuous"
)

def is_classic_control_env(env: Union[str, gym.Env]) -> bool:
    if isinstance(env, str) and env.startswith(classic_control_env_prefixes):
        return True
    if isinstance(env, gym.Env) and isinstance(env.unwrapped, classic_control_envs):
        return True
    return False

def is_atari_env(env: Union[str, gym.Env]) -> bool:
    # TODO: Add more names from the atari environments, or figure out a smarter
    # way to do this.
    if isinstance(env, str) and env.startswith("Breakout"):
        return True
    try:
        from gym.envs.atari import AtariEnv
        return isinstance(env, gym.Env) and isinstance(env.unwrapped, AtariEnv)
    except gym.error.DependencyNotInstalled:
        return False
    return False

logger = get_logger(__file__)

EnvType = TypeVar("EnvType", bound=gym.Env)
ObservationType = TypeVar("ObservationType")
ActionType = TypeVar("ActionType")
RewardType = TypeVar("RewardType")



class StepResult(NamedTuple, Generic[ObservationType, RewardType]):
    state: ObservationType
    reward: RewardType
    done: bool
    info: Dict
    

def has_wrapper(env: gym.Wrapper, wrapper_type: Type[gym.Wrapper]) -> bool:
    """Returns wether the given `env` has a wrapper of type `wrapper_type`. 

    Args:
        env (gym.Wrapper): a gym.Wrapper or a gym environment.
        wrapper_type (Type[gym.Wrapper]): A type of Wrapper to check for.

    Returns:
        bool: Wether there is a wrapper of that type wrapping `env`. 
    """
    # avoid cycles, although that would be very weird to encounter.
    while hasattr(env, "env") and env.env is not env:
        if isinstance(env, wrapper_type):
            return True
        env = env.env
    return isinstance(env, wrapper_type)


from .env_dataset import EnvDataset
from .policy_env import PolicyEnv

from abc import ABC


class MayCloseEarly(gym.Wrapper, ABC):
    """ ABC for Wrappers that may close an environment early depending on some
    conditions.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._is_closed: bool = False

    def is_closed(self) -> bool:
        # First, make sure that we're not 'overriding' the 'is_closed' of the
        # wrapped environment.
        if hasattr(self.env, "is_closed"):
            assert callable(self.env.is_closed)
            self._is_closed = self.env.is_closed()
        return self._is_closed

    def close(self) -> None:
        self.env.close()
        self._is_closed = True


class IterableWrapper(MayCloseEarly, IterableDataset, Generic[EnvType], ABC):
    """ ABC that allows iterating over the wrapped env, if it is iterable.
    
    This allows us to wrap dataloader-based Environments and still use the gym
    wrapper conventions.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
    
    def __next__(self):
        # TODO: This is tricky. We want the wrapped env to use *our* step,
        # reset(), action(), observation(), reward() methods, instead of its own!
        # Otherwise if we are transforming observations for example, those won't
        # be affected. 
        # logger.debug(f"Wrapped env {self.env} isnt a PolicyEnv or an EnvDataset")
        # return type(self.env).__next__(self)
        return self.batch(next(self.env))
        
        if has_wrapper(self.env, EnvDataset):
            logger.debug(f"Wrapped env is an EnvDataset, using EnvDataset.__next__.")
            return EnvDataset.__next__(self)
        # return self.observation(obs)
    
    def reset(self):
        return self.observation(self.env.reset())
    
    def step(self, action):
        action = self.action(action)
        observation, reward, done, info = self.env.step(action)
        observation = self.observation(observation)
        reward = self.reward(reward)
        return observation, reward, done, info

    def observation(self, observation):
        return observation

    def action(self, action):
        return action

    def reward(self, reward):
        return reward

    def batch(self, batch):
        """ Transform to be applied to the items yielded when iterating over the env.

        Defaults to just doing the same as the transform for the observations.
        """
        return self.observation(batch)

    def __len__(self):
        return self.env.__len__()

    def send(self, action):
        action = self.action(action)
        reward = self.env.send(action)
        reward = self.reward(reward)
        return reward

    def __iter__(self) -> Iterator:
        for batch in self.env:
            yield self.batch(batch)

    # def __setattr__(self, attr, value):
    #     """ Redirect the __setattr__ of attributes 'owned' by the EnvDataset to
    #     the EnvDataset.
        
    #     We need to do this because we change the value of `self` and call
    #     EnvDataset.__iter__(self), which might get and set attributes to/from
    #     `self`, which is what you'd expect normally. However when `self` is a
    #     wrapper over the env, rather than the env itself, then when attributes
    #     are set on `self` inside __iter__ or __next__ or send, etc, they are
    #     actually set on the wrapper, rather than on the env.
        
    #     We solve this by detecting when an attribute with a name ending with "_"
    #     and part of a given list of attributes is set.
    #     """
    #     if attr.endswith("_") and has_wrapper(self.env, EnvDataset):
    #         if attr in {"observation_", "action_", "reward_", "done_", "info_", "n_sends_"}:
    #             # logger.debug(f"Attribute {attr} will be set on the wrapped env rather than on the wrapper itself.")
    #             env = self.env
    #             while not isinstance(env, EnvDataset) and env.env is not env:
    #                 env = env.env
    #             assert isinstance(env, EnvDataset)
    #             setattr(env, attr, value)
    #     else:
    #         object.__setattr__(self, attr, value)


class RenderEnvWrapper(IterableWrapper):
    """ Simple Wrapper that renders the env at each step. """

    def step(self, action):
        self.env.render("human")
        return self.env.step(action)


@singledispatch
def reshape_space(space: gym.Space, new_shape: Tuple[int, ...]) -> gym.Space:
    """ Returns a new space based on 'space', but with a new shape.
    The space might change type, for instance Discrete(2) with new shape (3,)
    will become Tuple(Discrete(2), Discrete(2), Discrete(2)).
    """
    if isinstance(space, spaces.Space):
        # Space is of some other type. Hope that the shapes are the same.
        if new_shape == space.shape:
            return space
    raise NotImplementedError(f"Don't know how to reshape space {space} to have new shape {new_shape}")

@reshape_space.register
def reshape_box(space: spaces.Box, new_shape: Tuple[int, ...]) -> spaces.Box:
    assert isinstance(new_shape, (tuple, list))
    # TODO: For now just assume that all the bounds are the same value.
    low = space.low.reshape(new_shape)
    high = space.high.reshape(new_shape)
    return type(space)(low=low, high=high, dtype=space.dtype)

@reshape_space.register
def reshape_discrete(space: spaces.Discrete, new_shape: Tuple[int, ...]) -> spaces.Discrete:
    # Can't change the shape of a Discrete space, return a new one anyway.
    assert space.shape == (), "Discrete spaces should have empty shape."
    assert new_shape in [(), None], f"Can't change the shape of a Discrete space to {new_shape}."
    return spaces.Discrete(n=space.n)

@reshape_space.register
def reshape_tuple(space: spaces.Tuple, new_shape: Tuple[int, ...]) -> spaces.Tuple:
    assert isinstance(new_shape, (tuple, list))
    assert len(new_shape) == len(space), "Need len(new_shape) == len(space.spaces)"
    return spaces.Tuple([
        reshape_space(space_i, shape_i)
        for (space_i, shape_i) in zip(space.spaces, new_shape)
    ])

@reshape_space.register
def reshape_dict(space: spaces.Dict, new_shape: Tuple[int, ...]) -> spaces.Dict:
    assert isinstance(new_shape, dict) or len(new_shape) == len(space)
    return spaces.Dict({
        k: reshape_space(v, new_shape[k if isinstance(new_shape, dict) else i])
        for i, (k, v) in enumerate(space.spaces.items())
    })

