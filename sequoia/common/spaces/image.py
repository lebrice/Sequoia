""" IDEA: Create a subclass of spaces.Box for images.
"""
from typing import Optional, Tuple, Union

import numpy as np
from gym import spaces
from gym.vector.utils import batch_space

import torch

from .tensor_spaces import TensorBox
from .space import Space, T


class Image(spaces.Box, Space[T]):
    """ Subclass of `gym.spaces.Box` for images.

    Comes with a few useful attributes, like `h`, `w`, `c`, `channels_first`,
    `channels_last`, etc.
    """

    def __init__(
        self,
        low: Union[float, np.ndarray],
        high: Union[float, np.ndarray],
        shape: Tuple[int, ...] = None,
        dtype: np.dtype = None,
        **kwargs
    ):
        if dtype is None:
            if (
                isinstance(low, int)
                and isinstance(high, int)
                and low == 0
                and high == 255
            ):
                dtype = np.uint8
            else:
                dtype = np.float32
        super().__init__(low=low, high=high, shape=shape, dtype=dtype, **kwargs)
        self.channels_first: bool = False

        # Optional batch dimension
        self.b: Optional[int] = None
        self.h: int
        self.w: int
        self.c: int
        assert len(self.shape) in {3, 4}, "Need three or four dimensions."
        if len(self.shape) == 3:
            self.b = None
            if self.shape[0] in {1, 3}:
                self.c, self.h, self.w = self.shape
                self.channels_first = True
            elif self.shape[-1] in {1, 3}:
                self.h, self.w, self.c = self.shape
            else:
                # NOTE: will assume that in channels_first for now, but won't set
                # `channels_first` property.
                self.c, self.h, self.w = self.shape
        elif len(self.shape) == 4:
            if self.shape[1] in {1, 3}:
                self.b, self.c, self.h, self.w = self.shape
                self.channels_first = True
            elif self.shape[-1] in {1, 3}:
                self.b, self.h, self.w, self.c = self.shape
            else:
                # NOTE: will assume that in channels_first for now:
                self.b, self.c, self.h, self.w = self.shape

        if any(v is None for v in [self.h, self.w, self.c]):
            raise RuntimeError(
                f"Shouldn't be using an Image space, since the shape "
                f"doesn't appear to be an image: {self.shape}"
            )

    @property
    def channels(self) -> int:
        return self.c

    @property
    def height(self) -> int:
        return self.h

    @property
    def width(self) -> int:
        return self.w

    @property
    def batch_size(self) -> Optional[int]:
        return self.b

    @classmethod
    def from_box(cls, box_space: spaces.Box):
        return cls(box_space.low, box_space.high, dtype=box_space.dtype)

    @classmethod
    def wrap(cls, space: Union["Image", spaces.Box]):
        if isinstance(space, Image):
            return space
        if isinstance(space, spaces.Box):
            return cls.from_box(space)
        raise NotImplementedError(space)

    @property
    def channels_last(self) -> bool:
        return not self.channels_first

    def __repr__(self):
        return f"{type(self).__name__}({self.low.min()}, {self.high.max()}, {self.shape}, {self.dtype})"

    def sample(self) -> T:
        return super().sample()


class ImageTensorSpace(Image, TensorBox):
    @classmethod
    def from_box(cls, box_space: TensorBox, device: torch.device=None):
        device = device or box_space.device
        return cls(box_space.low, box_space.high, dtype=box_space.dtype, device=device)

    def __repr__(self):
        return f"{type(self).__name__}({self.low.min()}, {self.high.max()}, {self.shape}, {self.dtype}, device={self.device})"

    def sample(self):
        self.dtype = self._numpy_dtype
        s = super().sample()
        self.dtype = self._torch_dtype
        return torch.as_tensor(s, dtype=self._torch_dtype, device=self.device)


# @to_tensor.register
# def _(space: Image,
#       sample: Union[np.ndarray, Tensor],
#       device: torch.device = None) -> Union[Tensor]:
#     """ Converts a sample from the given space into a Tensor. """
#     return torch.as_tensor(sample, device=device)


@batch_space.register
def _batch_image_space(space: Image, n: int = 1) -> Union[Image, spaces.Box]:
    if space.b is not None:
        # This might happen in BatchedVectorEnv, when creating env_a and env_b,
        # which have an extra batch/chunk dimension.
        if space.b == 1:
            if n == 1:
                return space
            repeats = [n, 1, 1, 1]
        else:
            # instead maybe we should just fall back to a Box Space?
            repeats = [n] + [1] * space.low.ndim
            low, high = np.tile(space.low, repeats), np.tile(space.high, repeats)
            return spaces.Box(low=low, high=high, dtype=space.dtype)

            raise RuntimeError(
                f"can't batch an already batched image space {space}, n={n}"
            )
    else:
        repeats = [n, 1, 1, 1]
    low, high = np.tile(space.low, repeats), np.tile(space.high, repeats)
    img = type(space)(low=low, high=high, dtype=space.dtype)
    return img
