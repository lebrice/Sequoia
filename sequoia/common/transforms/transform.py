""" Defines a 'smarter' Transform class. """
from typing import overload
from abc import ABC, abstractmethod
from typing import (Any, Callable, Generic, Sized, Tuple, TypeVar, Union,
                    overload)
import warnings

import gym
import numpy as np
from gym import spaces, Space
from torch import Tensor
from PIL.Image import Image

InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")

Img = TypeVar("Img", bound=Union[Image, np.ndarray, Tensor])
Shape = TypeVar("Shape", bound=Tuple[int, ...])

class Transform(Generic[InputType, OutputType]):
    """ Callable that can also tell you its impact on the shape of inputs. """

    @overload
    def __call__(self, input: InputType) -> OutputType:
        ...

    @overload
    def __call__(self, input: Shape) -> Shape:
        ...

    @overload
    def __call__(self, input: Space) -> Space:
        ...

    @abstractmethod
    def __call__(self, input: Union[InputType, Space, Shape]) -> Union[OutputType, Space, Shape]:
        pass

    # def shape_change(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    #     """ Gives the impact this transform would have on the shape of an input.

    #     NOTE: Maybe later if some transforms create tuples, like SIMCLR, or if
    #     they also create labels (like dicts) or somethings, then we probably
    #     will have to change this.
    #     """
    #     warnings.warn(DeprecationWarning("Don't call shape_change, instead apply the transform to a shape directly."))
    #     raise NotImplementedError(f"TODO: Remove this and add Space support to {self}.")
    #     # Default to saying that this transform doesn't affect the shape.
    #     return input_shape

    # def space_change(self, input_space: gym.Space) -> gym.Space:
    #     """ Gives the impact this transform would have on an input gym.Space.
    #     """
    #     warnings.warn(DeprecationWarning("Don't call space_change, instead apply the transform to a Space directly."))
    #     raise NotImplementedError(f"TODO: Remove this and add Space support to {self}.")

        # def _get_shape(space: gym.Space) -> Tuple:
        #     if isinstance(space, spaces.Box):
        #         return space.shape
        #     if isinstance(input_space, spaces.Tuple):
        #         return tuple(map(_get_shape, space.spaces))
        #     if isinstance(input_space, spaces.Dict):
        #         return tuple(map(_get_shape, space.spaces.values()))
        #     raise NotImplementedError(f"Dont know how to get the shape of space {space}.")
        # input_shape = _get_shape(input_space)
        # output_shape = self.shape_change(input_shape)

        # from sequoia.common.gym_wrappers.utils import reshape_space
        # return reshape_space(input_space, output_shape)
