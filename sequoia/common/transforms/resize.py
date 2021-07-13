from functools import singledispatch
from typing import Any, Callable, Sequence, Tuple, TypeVar, Union, List, Dict

import gym
import numpy as np
import torch
from gym import spaces
from PIL import Image
from sequoia.utils.logging_utils import get_logger
from torch import Tensor
from torch.nn.functional import interpolate
from torchvision.transforms import InterpolationMode
from torchvision.transforms import Resize as Resize_
from torchvision.transforms import ToTensor as ToTensor_
from torchvision.transforms import functional as F

from .channels import (
    channels_first,
    channels_last,
    has_channels_first,
    has_channels_last,
)
from .transform import Img, Transform
from sequoia.common.spaces import NamedTupleSpace, TypedDictSpace
from sequoia.common.spaces.image import Image as ImageSpace
from sequoia.common.gym_wrappers.convert_tensors import (
    has_tensor_support,
    add_tensor_support,
)
from collections.abc import Mapping
from sequoia.settings.base import Observations
from .utils import is_image

logger = get_logger(__file__)


@singledispatch
def resize(x: Img, size: Tuple[int, ...], **kwargs) -> Img:
    """ Resizes a PIL.Image, a Tensor, ndarray, or a Box space. """
    raise NotImplementedError(f"Transform doesn't support input {x} of type {type(x)}")


@resize.register
def _(x: Image.Image, size: Tuple[int, ...], **kwargs) -> Image.Image:
    return F.resize(x, size, **kwargs)


@resize.register(np.ndarray)
@resize.register(Tensor)
def _resize_array_or_tensor(
    x: np.ndarray, size: Tuple[int, ...], **kwargs
) -> np.ndarray:
    """ TODO: This resizes numpy arrays by converting them to tensors and then
    using the `interpolate` function. There is for sure a more efficient way to
    do this.
    """
    original = x
    if isinstance(original, np.ndarray):
        # Need to convert to tensor (for interpolate to work).
        x = torch.as_tensor(x)
    if len(original.shape) == 3:
        # Need to add a batch dimension (for interpolate to work).
        x = x.unsqueeze(0)
    if has_channels_last(original):
        # Need to make it channels first (for interpolate to work).
        x = channels_first(x)

    assert has_channels_first(
        x
    ), f"Image needs to have channels first (shape is {x.shape})"

    x = interpolate(x, size, mode="area")
    if isinstance(original, np.ndarray):
        x = x.numpy()
    if len(original.shape) == 3:
        x = x[0]
    if has_channels_last(original):
        x = channels_last(x)
    return x


@resize.register
def _resize_namedtuple_space(
    x: NamedTupleSpace, size: Tuple[int, ...], **kwargs
) -> NamedTupleSpace:
    """ When presented with a NamedTupleSpace input, this transform will be
    applied to all 'Image' spaces.
    """
    return type(x)(
        **{
            key: resize(v, size, **kwargs) if isinstance(v, ImageSpace) else v
            for key, v in x._spaces.items()
        }
    )


@resize.register(Mapping)
def _resize_namedtuple(x: Dict, size: Tuple[int, ...], **kwargs) -> Dict:
    """ When presented with a Mapping-like input, this transform will be
    applied to all 'Image' spaces.
    """
    return type(x)(
        **{
            key: resize(value, size, **kwargs) if is_image(value) else value
            for key, value in x.items()
        }
    )


@resize.register(TypedDictSpace)
def _resize_typed_dict(x: TypedDictSpace, size: Tuple[int, ...], **kwargs) -> TypedDictSpace:
    """ When presented with a Mapping-like input, this transform will be
    applied to all 'Image' spaces.
    """
    return type(x)(
        {
            key: resize(value, size, **kwargs) if is_image(value) else value
            for key, value in x.items()
        }, dtype=x.dtype,
    )


@resize.register(tuple)
def _resize_image_shape(
    x: Tuple[int, ...], size: Tuple[int, ...], **kwargs
) -> Tuple[int, ...]:
    """ Give the resized image shape, given the input shape. """
    new_shape: List[int] = list(size)
    if len(size) == 2:
        # Preserve the number of channels.
        if len(x) == 4:
            if has_channels_first(x):
                new_shape = [*x[:2], *size]
            elif has_channels_last(x):
                new_shape = [x[0], *size, x[-1]]
            else:
                raise NotImplementedError(x)
        elif len(x) == 3:
            if has_channels_first(x):
                new_shape = [x[0], *size]
            elif has_channels_last(x):
                new_shape = [*size, x[-1]]
            else:
                raise NotImplementedError(x)
    else:
        NotImplementedError(size)
    return type(x)(new_shape)


@resize.register(spaces.Box)
def _resize_space(x: spaces.Box, size: Tuple[int, ...], **kwargs) -> spaces.Box:
    # Hmm, not sure if the bounds would actually also be respected though.
    new_space = type(x)(
        low=resize(x.low, size, **kwargs),
        high=resize(x.high, size, **kwargs),
        dtype=x.dtype,
    )
    # If the 'old' space supported tensors as samples, then so will the new space.
    if has_tensor_support(x):
        return add_tensor_support(new_space)
    return new_space


class Resize(Resize_, Transform[Img, Img]):
    def __init__(self, size: Tuple[int, ...], interpolation=InterpolationMode.BILINEAR):
        super().__init__(size, interpolation)
        # self.size = size
        # self.interpolation = interpolation

    def __call__(self, img):
        # TODO: (@lebrice) Weirdly enough, it seems that even though we
        # implement forward below, and __call__ is supposed to just use
        # `forward`, the base class somehow doesn't use our implementation, so
        # the test
        # env_dataset_test.py::test_iteration_with_more_than_one_wrapper would
        # fail if we don't have this __call__ explicitly implemented,
        return self.forward(img)

    def forward(self, img: Img) -> Img:
        return resize(img, size=self.size)
