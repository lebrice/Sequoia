""" Defines a 'smarter' Transform class. """
from abc import abstractmethod
from typing import Generic, Tuple, TypeVar, Union, overload

import numpy as np
from gym import Space
from PIL.Image import Image
from torch import Tensor

InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")

Img = TypeVar("Img", Image, np.ndarray, Tensor)
Shape = TypeVar("Shape", bound=Tuple[int, ...])


class Transform(Generic[InputType, OutputType]):
    """Callable that can also tell you its impact on the shape of inputs."""

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
