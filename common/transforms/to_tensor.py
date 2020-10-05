""" Slight modification of the ToTensor transform from TorchVision.

@lebrice: I wrote this because I would often get weird 'negative stride in
images' errors when converting PIL images from some gym environments when
using `ToTensor` from torchvision.
"""
from dataclasses import dataclass
from typing import Callable, Tuple, Union

import numpy as np
import torch
from PIL.Image import Image
from torch import Tensor
from torchvision.transforms import ToTensor as ToTensor_


def copy_if_negative_strides(image: Union[Image, np.ndarray, Tensor]):
    # It sometimes happens when taking images from a gym env that the strides
    # are negative, for some reason. Therefore we need to copy the array
    # before we can call torch.to_tensor(pic).

    if isinstance(image, Image):
        image = np.array(image)
    if isinstance(image, np.ndarray):
        strides = image.strides
    elif isinstance(image, Tensor):
        strides = image.stride()
    if any(s < 0 for s in strides):
        return image.copy()
    return image

from torchvision.transforms import functional as F


def to_tensor(pic) -> Tensor:
    if isinstance(pic, Tensor):
        return pic
    pic = copy_if_negative_strides(pic)
    if len(pic.shape) == 4:
        return torch.stack([F.to_tensor(p) for p in pic])
    return F.to_tensor(pic)


@dataclass
class ToTensor(ToTensor_):

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