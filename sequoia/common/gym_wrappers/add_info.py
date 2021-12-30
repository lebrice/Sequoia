""" Wrapper that adds the 'info' as a part of the environment's observations.
"""
from dataclasses import dataclass, is_dataclass, replace
from functools import singledispatch
from typing import Any, Dict, Sequence, Tuple, TypeVar, Union

import gym
import numpy as np
from gym import Space, spaces
from gym.vector import VectorEnv
from sequoia.common.spaces.utils import batch_space
from torch import Tensor

from .utils import IterableWrapper, has_wrapper

Info = TypeVar("Info", bound=Union[Dict, Sequence[Dict]])
K = TypeVar("K")
V = TypeVar("V")


@singledispatch
def add_info(observation, info):
    """ Generic function that adds the provided `info` value to an observation.
    Returns the modified observation, which might not always be of the same type.

    NOTE: Can also be applied to spaces. 
    """
    if is_dataclass(observation):
        # TODO: This assumes that the dataclass already has the 'info' field, if
        # that dataclass is frozen.
        return replace(observation, info=info)
    raise NotImplementedError(
        f"Function add_info has no handler registered for inputs of type "
        f"{type(observation)}."
    )


@add_info.register(Tensor)
@add_info.register(np.ndarray)
def _add_info_to_array_obs(observation: np.ndarray, info: Info) -> Tuple[np.ndarray, Info]:
    return (observation, info)


@add_info.register(tuple)
def _add_info_to_tuple_obs(observation: Tuple, info: Info) -> Tuple:
    return observation + (info,)


@add_info.register(dict)
def _add_info_to_dict_obs(observation: Dict[K, V], info: Info) -> Dict[K, Union[V, Info]]:
    assert "info" not in observation
    observation["info"] = info
    return observation


@add_info.register(spaces.Space)
def add_info_to_space(observation: Space, info: Space) -> Space:
    """ Adds the space of the 'info' value from the env to this observation
    space.
    """
    raise NotImplementedError(
        f"No handler registered for spaces of type {type(observation)}. "
        f"(value = {observation})"
    )


@add_info.register
def _add_info_to_box_space(observation: spaces.Box, info: Space) -> spaces.Tuple:
    return spaces.Tuple([
        observation,
        info,
    ])


@add_info.register
def _add_info_to_tuple_space(observation: spaces.Tuple, info: Space) -> spaces.Tuple:
    return spaces.Tuple([
        *observation.spaces,
        info,
    ])


@add_info.register
def _add_info_to_dict_space(observation: spaces.Dict, info: Space) -> spaces.Dict:
    new_spaces = observation.spaces.copy()
    assert "info" not in new_spaces, "space shouldn't already have an 'info' key."
    new_spaces["info"] = info
    return type(observation)(new_spaces)


class AddInfoToObservation(IterableWrapper):
    # TODO: Need to add the 'info' dict to the Observation, so we can have
    # access to the final observation (which gets stored in the info dict at key
    # 'final_state'.
    # Do we through?
    
    # TODO: Should we also add the 'final state' to the observations as well?

    def __init__(self,
                 env: gym.Env,
                 info_space: spaces.Space = None):
        super().__init__(env)
        self.is_vectorized = isinstance(env.unwrapped, VectorEnv)
        # TODO: Should we make 'info_space' mandatory here?
        if info_space is None:
            # TODO: There seems to be some issues if we have an empty info space
            # before the batching.
            info_space = spaces.Dict({})
            if self.is_vectorized:
                info_space = batch_space(info_space, self.env.num_envs)
        self.info_space = info_space
        self.observation = add_info(self.env.observation_space, self.info_space)

    def reset(self, **kwargs):
        observation = self.env.reset()
        info = {}
        if self.is_vectorized:
            info = np.array([{} for _ in range(self.env.num_envs)])
        obs = add_info(observation, info)
        return obs

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = add_info(observation, info)
        return observation, reward, done, info
