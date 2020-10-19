""" Slight modification of the ToTensor transform from TorchVision.

@lebrice: I wrote this because I would often get weird 'negative stride in
images' errors when converting PIL images from some gym environments when
using `ToTensor` from torchvision.
"""
from dataclasses import dataclass
from typing import Callable, Tuple, Union, TypeVar

import gym
import numpy as np
import torch
from gym import spaces
from PIL.Image import Image
from torch import Tensor
from torchvision.transforms import ToTensor as ToTensor_
from .transform import Transform

T = TypeVar("T", bound=Union[Image, np.ndarray, Tensor])


def copy_if_negative_strides(image: T) -> T:
    # It sometimes happens when taking images from a gym env that the strides
    # are negative, for some reason. Therefore we need to copy the array
    # before we can call torch.to_tensor(pic).
    if isinstance(image, Image):
        image = np.array(image)

    if isinstance(image, np.ndarray):
        strides = image.strides
    elif isinstance(image, Tensor):
        strides = image.stride()
    elif hasattr(image, "strides"):
        strides = image.strides
    else:
        # Can't get strides of object, return it unchanged?
        return image
    if any(s < 0 for s in strides):
        return image.copy()
    return image

from torchvision.transforms import functional as F


def to_tensor(pic) -> Tensor:
    tensor: Tensor
    if isinstance(pic, Tensor):
        tensor = pic
    else:
        pic = copy_if_negative_strides(pic)
        if len(pic.shape) == 4:
            tensor = torch.stack([F.to_tensor(p) for p in pic])
        else:
            tensor = F.to_tensor(pic)
    return tensor


@dataclass
class ToTensor(ToTensor_, Transform):
    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, Tensor):
            return pic
        pic = copy_if_negative_strides(pic)
        return super().__call__(pic)

    def shape_change(self, input_shape: Union[Tuple[int, ...], torch.Size]) -> Tuple[int, ...]:
        return input_shape

    def space_change(self, input_space: gym.Space) -> gym.Space:
        return spaces.Box(
            low=0.,
            high=1.,
            shape=input_space.shape,
            dtype=np.float32,
        )
        