from abc import ABC
from typing import (Dict, Generic, Iterator, List, NamedTuple, Tuple, Type,
                    TypeVar)
from collections.abc import Sized
from functools import singledispatch
import gym
import numpy as np
from gym import spaces
from gym.vector.utils import batch_space
from torch.utils.data import IterableDataset

from utils.logging_utils import get_logger


from gym.envs.classic_control import AcrobotEnv, CartPoleEnv, PendulumEnv, MountainCarEnv, Continuous_MountainCarEnv
classic_control_envs = (AcrobotEnv, CartPoleEnv, PendulumEnv, MountainCarEnv, Continuous_MountainCarEnv)

classic_control_env_prefixes: Tuple[str, ...] = (
    "CartPole", "Pendulum", "Acrobot", "MountainCar", "MountainCarContinuous"
)


logger = get_logger(__file__)

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
    return False


def remove_wrapper(env: gym.Wrapper, wrapper_type: Type[gym.Wrapper]) -> gym.Wrapper:
    """ IDEA: remove a given wrapper. """
    raise NotImplementedError


class IterableWrapper(gym.Wrapper, IterableDataset, ABC):
    """ ABC that allows iterating over the wrapped env, if it is iterable.
    
    This allows us to wrap dataloader-based Environments and still use the gym
    wrapper conventions.
    """
    def __next__(self):
        return self.env.__next__()
    def __iter__(self) -> Iterator:
        return self.env.__iter__()


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
    low = space.low if np.isscalar(space.low) else next(space.low.flat)
    high = space.high if np.isscalar(space.high) else next(space.high.flat)
    return spaces.Box(low=low, high=high, shape=new_shape)

@reshape_space.register
def reshape_discrete(space: spaces.Discrete, new_shape: Tuple[int, ...]) -> spaces.Discrete:
    # Can't change the shape of a Discrete space, return a new one anyway.
    assert space.shape is (), "Discrete spaces should have empty shape."
    assert len(new_shape) == 0, f"Can't change the shape of a Discrete space to {new_shape}."
    return spaces.Discrete(n=space.n, dtype=space.dtype)

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

