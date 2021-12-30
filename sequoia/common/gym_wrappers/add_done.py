""" Wrapper that adds 'done' as part of the environment's observations.
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
from sequoia.common.spaces import TypedDictSpace

from sequoia.common.spaces import TypedDictSpace

from .utils import IterableWrapper, has_wrapper

T = TypeVar("T")
Bool = TypeVar("Bool", bound=Union[bool, Sequence[bool]])
K = TypeVar("K")
V = TypeVar("V")


@singledispatch
def add_done(observation: Any, done: Any) -> Any:
    """ Generic function that adds the provided `done` value to an observation.
    Returns the modified observation, which might not always be of the same type.
    """
    if is_dataclass(observation):
        return replace(observation, done=done)
    raise NotImplementedError(
        f"Function add_done has no handler registered for observations of type "
        f"{type(observation)}."
    )

@add_done.register(int)
@add_done.register(float)
@add_done.register(Tensor)
@add_done.register(np.ndarray)
def _add_done_to_array_obs(observation: T, done: bool) -> Dict[str, Union[T, bool]]:
    # TODO: use 'x' or 'observation'?
    return {"x": observation, "done": done}


@add_done.register(tuple)
def _add_done_to_tuple_obs(observation: Tuple, done: bool) -> Tuple:
    return observation + (done,)


@add_done.register(dict)
def _add_done_to_dict_obs(observation: Dict[K, V], done: bool) -> Dict[K, Union[V, bool]]:
    assert "done" not in observation
    observation["done"] = done
    return observation


@add_done.register
def add_done_to_space(observation: Space, done: Space) -> Space:
    """ Adds the space of the 'done' value to the given space.
    
    By default, `done` corresponds to what you'd get from a single
    (i.e. non-vectorized) environment. 
    """
    raise NotImplementedError(
        f"No handler registered for spaces of type {type(observation)}. "
        f"(value = {observation}, done={done})"
    )


@add_done.register(spaces.Discrete)
@add_done.register(spaces.MultiDiscrete)
@add_done.register(spaces.MultiBinary)
@add_done.register(spaces.Box)
def _add_done_to_box_space(observation: Space, done: Space) -> spaces.Dict:
    # TODO: Use 'x' or 'observation' as the key?
    return TypedDictSpace(
        x=observation,
        done=done,
    )


@add_done.register
def _add_done_to_tuple_space(observation: spaces.Tuple, done: Space) -> spaces.Tuple:
    return spaces.Tuple([
        *observation.spaces,
        done,
    ])


@add_done.register
def _add_done_to_dict_space(observation: spaces.Dict, done: Space) -> spaces.Dict:
    new_spaces = observation.spaces.copy()
    assert "done" not in new_spaces, "space shouldn't already have a 'done' key."
    new_spaces["done"] = done
    return type(observation)(new_spaces)


class AddDoneToObservation(IterableWrapper):
    """Wrapper that adds the 'done' from step to the 
    Need to add the 'done' vector to the observation, so we can
    get access to the 'end of episode' signal in the shared_step, since
    when iterating over the env like a dataloader, the yielded items only
    have the observations, and dont have the 'done' vector. (so as to be
    consistent with supervised learning).
    
    NOTE: NEVER use this *BEFORE* batching, because of how the 'reset' works in
    all VectorEnvs, the observations will always be the 'new' ones, so `done`
    (in the obs) will always be False!
    """
    def __init__(self, env: gym.Env, done_space: Space = None):
        super().__init__(env)
        # boolean value. (0 or 1)
        if done_space is None:
            done_space = spaces.Box(0, 1, (), dtype=np.bool)
            if self.is_vectorized:
                self.single_observation_space = add_done(self.single_observation_space, done_space)
                done_space = batch_space(done_space, self.env.num_envs)
        self.done_space = done_space
        self.observation_space = add_done(self.env.observation_space, self.done_space)


    def reset(self, **kwargs):
        observation = self.env.reset()
        if self.is_vectorized:
            done = self.done_space.low
        else:
            done = False
        return add_done(observation, done)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = add_done(observation, done)
        return observation, reward, done, info
