""" Slight modification of the ToTensor transform from TorchVision.

@lebrice: I wrote this because I would often get weird 'negative stride in
images' errors when converting PIL images from some gym environments when
using `ToTensor` from torchvision.
"""
from dataclasses import dataclass
from typing import Callable, Tuple, Union, TypeVar, Sequence

import gym
import numpy as np
import torch
from gym import spaces
from PIL.Image import Image
from torch import Tensor
from torchvision.transforms import ToTensor as ToTensor_
from torchvision.transforms import functional as F

from utils.logging_utils import get_logger

from .channels import ChannelsFirstIfNeeded, ChannelsLastIfNeeded
from .transform import Transform, Img

logger = get_logger(__file__)

channels_first = ChannelsFirstIfNeeded()
channels_last = ChannelsLastIfNeeded()



def copy_if_negative_strides(image: Img) -> Img:
    # It sometimes happens when taking images from a gym env that the strides
    # are negative, for some reason. Therefore we need to copy the array
    # before we can call torchvision.transforms.functional.to_tensor(pic).
    if isinstance(image, Image):
        image = np.array(image)

    if isinstance(image, np.ndarray):
        strides = image.strides
    elif isinstance(image, Tensor):
        strides = image.stride()
    elif hasattr(image, "strides"):
        strides = image.strides
    else:
        raise NotImplementedError(f"Can't get strides of object {image}")
    if any(s < 0 for s in strides):
        return image.copy()
    return image


def to_tensor(pic: Union[Img, Sequence[Img]]) -> Tensor:
    """
    Converts a PIL Image or numpy.ndarray ((N) x H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape ((N) x C x H x W) in the range
    [0.0, 1.0] if the PIL Image belongs to one of the modes (L, LA, P, I, F,
    RGB, YCbCr, RGBA, CMYK, 1) or if the numpy.ndarray has dtype = np.uint8

    Parameters
    ----------
    pic : Union[Img, Sequence[Img]]
        [description]

    Returns
    -------
    Tensor
        [description]
    """
    tensor: Tensor
    if isinstance(pic, Tensor):
        return channels_first(pic)

    assert isinstance(pic, (np.ndarray, Image, list, tuple))
    
    if isinstance(pic, (list, tuple)) or (isinstance(pic, np.ndarray) and pic.ndim == 4):
        return torch.stack(list(map(to_tensor, pic)))

    assert isinstance(pic, (np.ndarray, Image))
    pic = copy_if_negative_strides(pic)

    if isinstance(pic, np.ndarray):
        # Convert to channels last if needed, because ToTensor expects to
        # receive that.
        if len(pic.shape) == 2:
            pass
        elif pic.shape[-1] not in {1, 3}:
            assert pic.shape[0] in {1, 3}, pic.shape
            pic = pic.transpose(1, 2, 0)
        # pic = channels_last(pic)
    pic = F.to_tensor(pic)
    assert isinstance(pic, Tensor), pic.shape
    return pic


@dataclass
class ToTensor(ToTensor_, Transform):
    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
        
        Returns:
            Tensor: Converted image.
        
        NOTE: torchvision's ToTensor transform assumes that whatever it is given
        is always in channels_last format (as is usually the case with PIL
        images) and always returns images with the channels last:
        
            Converts a PIL Image or numpy.ndarray (H x W x C) in the range
            [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
            if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
            or if the numpy.ndarray has dtype = np.uint8
        """
        t = to_tensor(pic)
        assert isinstance(t, Tensor), type(t)
        return t

    @classmethod
    def shape_change(cls, input_shape: Union[Tuple[int, ...], torch.Size]) -> Tuple[int, ...]:
        from .channels import ChannelsFirstIfNeeded
        return ChannelsFirstIfNeeded.shape_change(input_shape)

    @classmethod
    def space_change(cls, input_space: gym.Space) -> gym.Space:
        if not isinstance(input_space, spaces.Box):
            logger.warning(UserWarning(f"Transform {cls} is only meant for Box spaces, not {input_space}"))
            return input_space
        return spaces.Box(
            low=0.,
            high=1.,
            shape=cls.shape_change(input_space.shape),
            dtype=np.float32,
        )
        