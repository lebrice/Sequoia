""" Defines a 'smarter' Transform class. """
from abc import ABC, abstractmethod
from typing import (Any, Callable, Generic, Sized, Tuple, TypeVar, Union,
                    overload)

import gym
from gym import spaces
from torch import Tensor

InputType = TypeVar("InputType", bound=Sized)
OutputType = TypeVar("OutputType", bound=Sized)


class Transform(Generic[InputType, OutputType]):
    """ Callable that can also tell you its impact on the shape of inputs. """

    @abstractmethod
    def __call__(self, input: InputType) -> OutputType:
        pass
    
    @abstractmethod
    def shape_change(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """ Gives the impact this transform would have on the shape of an input.
        
        NOTE: Maybe later if some transforms create tuples, like SIMCLR, or if
        they also create labels (like dicts) or somethings, then we probably
        will have to change this.

        TODO: To push this idea even further, we could change this so it also
        accepts a gym.Space, and return a gym.Space!
        """
        # Default to saying that this transform doesn't affect the shape.
        return input_shape
    
    def space_change(self, input_space: gym.Space) -> gym.Space:
        """ Gives the impact this transform would have on an input gym.Space.
        """
        def _get_shape(space: gym.Space) -> Tuple:
            if isinstance(space, spaces.Box):
                return space.shape
            if isinstance(input_space, spaces.Tuple):
                return tuple(map(_get_shape, space.spaces))
            if isinstance(input_space, spaces.Dict):
                return tuple(map(_get_shape, space.spaces.values()))
            raise NotImplementedError(f"Dont know how to get the shape of space {space}.")
        input_shape = _get_shape(input_space)
        output_shape = self.shape_change(input_shape)
        return space_with_new_shape(input_space, output_shape)

import numpy as np

def space_with_new_shape(space: gym.Space, new_shape: Tuple[int, ...]) -> gym.Space:
    """ Returns a new space of the same type, but with a new shape. """
    if isinstance(space, spaces.Box):
        assert isinstance(new_shape, (tuple, list))
        space: spaces.Box
        low = space.low if np.isscalar(space.low) else next(space.low.flat)
        high = space.high if np.isscalar(space.high) else next(space.high.flat)
        return spaces.Box(low=low, high=high, shape=new_shape)

    if isinstance(space, spaces.Discrete):
        # Can't change the shape of a Discrete space, return a new one anyway.
        assert space.shape is (), "Discrete spaces should have empty shape."
        assert new_shape is (), "Can't change the shape of a Discrete space."
        return spaces.Discrete(n=space.n)

    elif isinstance(space, spaces.Tuple):
        space: spaces.Tuple
        assert isinstance(new_shape, (tuple, list))
        assert len(new_shape) == len(space), "Need len(new_shape) == len(space.spaces)"
        return spaces.Tuple([
            space_with_new_shape(space_i, shape_i)
            for (space_i, shape_i) in zip(space.spaces, new_shape)
        ])
    elif isinstance(space, spaces.Dict):
        space: spaces.Dict
        assert isinstance(new_shape, dict) or len(new_shape) == len(space)
        return spaces.Dict({
            k: space_with_new_shape(v, new_shape[k if isinstance(new_shape, dict) else i])
            for i, (k, v) in enumerate(space.spaces.items())
        })
    elif isinstance(space, spaces.Space):
        # Space is of some other type. Hope that the shapes are the same.
        if new_shape == space.shape:
            return space
    raise NotImplementedError(
        f"Don't know how to change the shape of space {space} to {new_shape}. "
    )

