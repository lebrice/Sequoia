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

        from common.gym_wrappers.utils import reshape_space
        return reshape_space(input_space, output_shape)
