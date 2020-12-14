from typing import Callable, Sequence, Tuple, TypeVar, Union

import gym
import numpy as np
import torch
from gym import spaces
from PIL import Image
from torch import Tensor
from torchvision.transforms import Resize as Resize_
from torchvision.transforms import ToTensor as ToTensor_
from torch.nn.functional import interpolate

from sequoia.utils.logging_utils import get_logger
from .transform import Transform, Img
from .channels import has_channels_first, has_channels_last, ChannelsFirst, ChannelsLast

logger = get_logger(__file__)


class Resize(Resize_, Transform[Img, Img]):
    def __init__(self, size: Tuple[int, ...], interpolation=Image.BILINEAR):
        super().__init__(size, interpolation)

    def __call__(self, img: Img) -> Img:
        if not isinstance(img, Image.Image):
            original = img
            if isinstance(original, np.ndarray):
                img = torch.as_tensor(img)
            if len(original.shape) == 3:
                # Need to add a batch dimension.
                img = img.unsqueeze(0)
            if has_channels_last(original):
                # Need to make it channels first (for interpolate to work).
                img = ChannelsFirst.apply(img)
                
            assert has_channels_first(img), f"Image needs to have channels first (shape is {img.shape})"
            
            img = interpolate(img, self.size, mode="area")
            if isinstance(original, np.ndarray):
                img = img.numpy()
            if len(original.shape) == 3:
                img = img[0]
            if has_channels_last(original):
                img = ChannelsLast.apply(img)
            return img
        
        return super().__call__(img)

    def shape_change(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        assert isinstance(self.size, (list, tuple)) and len(self.size) == 2,  "Can only tell the space change from resize when size has two values atm."
        if has_channels_first(input_shape):
            *n, c, h, w = input_shape
            return (*n, c, *self.size)
        if has_channels_last(input_shape):
            *n, h, w, c = input_shape
            return (*n, *self.size, c)
        raise NotImplementedError(f"Don't know what the shape change is for input shape {input_shape}")
