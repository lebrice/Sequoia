""" IDEA: Create a subclass of spaces.Box for images.
"""
from gym import Space, spaces
from typing import Union, Tuple, Optional
import numpy as np


class Image(spaces.Box):
    """ Subclass of `gym.spaces.Box` for images.
    
    Comes with a few useful attributes, like `h`, `w`, `c`, `channels_first`,
    `channels_last`, etc.
    """
    def __init__(self,
                 low: Union[float, np.ndarray],
                 high: Union[float, np.ndarray],
                 shape: Tuple[int, ...] = None,
                 dtype: np.dtype = np.float32):
        super().__init__(low=low, high=high, shape=shape, dtype=dtype)
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
        elif len(self.shape) == 4:
            if self.shape[1] in {1, 3}:
                self.b, self.c, self.h, self.w = self.shape
                self.channels_first = True
            elif self.shape[-1] in {1, 3}:
                self.b, self.h, self.w, self.c = self.shape
        if any(v is None for v in [self.h, self.w, self.c]):
            raise RuntimeError(
                f"Shouldn't be using an Image space, since the shape "
                f"doesn't appear to be an image: {self.shape}"
            )

    @classmethod
    def from_box(cls, box_space: spaces.Box):
        return cls(box_space.low, box_space.high, dtype=box_space.dtype)

    @property
    def channels_last(self) -> bool:
        return not self.channels_first

    def __repr__(self):
        return f"Image({self.low.min()}, {self.high.max()}, {self.shape}, {self.dtype})"