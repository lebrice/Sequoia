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

Img = TypeVar("Img", Image, np.ndarray, Tensor)
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
