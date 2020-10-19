""" Transforms and such. Trying to make it possible to parse such from the
command-line.

Also, playing around with the idea of adding the ability to predict the change
in shape resulting from the transforms, à-la-Tensorflow.

"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, List, Tuple, TypeVar, Union

import gym
import numpy as np
import torch
from gym import spaces
from torch import Tensor
from torchvision.transforms import Compose as ComposeBase
from torchvision.transforms import RandomGrayscale
from torchvision.transforms import ToTensor as ToTensor_
from utils.logging_utils import get_logger
from utils.serialization import encode, register_decoding_fn

logger = get_logger(__file__)

from .channels import (ChannelsFirst, ChannelsFirstIfNeeded, ChannelsLast,
                       ChannelsLastIfNeeded, ThreeChannels)
from .to_tensor import ToTensor
from .transform import Transform
# TODO: Add names to the dimensions in the transforms!

# from pl_bolts.models.self_supervised.simclr import (SimCLREvalDataTransform,
#                                                     SimCLRTrainDataTransform)
class Transforms(Enum):
    """ Enum of possible transforms. 

    By having this as an Enum, we can choose which transforms to use from the
    command-line.
    This also makes it easier to check for identity, e.g. to check wether a
    particular transform was used.  

    TODO: Add the SimCLR/MOCO/etc transforms from  https://pytorch-lightning-bolts.readthedocs.io/en/latest/transforms.html
    TODO: Figure out a way to let people customize the arguments to the transforms?
    """
    three_channels = ThreeChannels()
    to_tensor = ToTensor()
    random_grayscale = RandomGrayscale()
    channels_first = ChannelsFirst()
    channels_first_if_needed = ChannelsFirstIfNeeded()
    channels_last = ChannelsLast()
    channels_last_if_needed = ChannelsLastIfNeeded()
    # simclr = Simclr

    def __call__(self, x):
        return self.value(x)

    @classmethod
    def _missing_(cls, value: Any):
        for e in cls:
            if e.name == value:
                return e
            elif type(e.value) == type(value):
                return e
        return super()._missing_(value)
    
    def shape_change(self, input_shape: Union[Tuple[int, ...], torch.Size]) -> Tuple[int, ...]:
        logger.debug(f"shape_change on the Transforms enum: self.value {self.value}, input shape: {input_shape}")
        # TODO: Give the impact of this transform on a given input shape.
        if hasattr(self.value, "shape_change"):
            output_shape = self.value.shape_change(input_shape)
            logger.debug(f"Output shape: {output_shape}")
            return output_shape
        # TODO: Maybe we could give it a random input of shape 'input_shape'
        # and check what kind of shape comes out of it? (This wouldn't work)
        # with things like PIL image transforms though.
        raise NotImplementedError("TODO")
        temp = torch.rand(input_shape)
        end = self.value(temp)
        return end.shape

    def space_change(self, input_space: gym.Space) -> gym.Space:
        return self.value.space_change(input_space)
    
# TODO: Add the SimCLR transforms.
# class SimCLRTrainTransform(SimCLRTrainDataTransform):
#     def __call

T = TypeVar("T", bound=Callable)

class Compose(List[T], ComposeBase):
    """ Extend the Compose class of torchvision with methods of `list`.
    
    This can also be passed in members of the `Transforms` enum, which makes it
    possible to do something like this:
    >>> from transforms import Compose, Transforms
    >>> transforms = Compose([Transforms.to_tensor, Transforms.fix_channels,])
    >>> Transforms.fix_channels in transforms
    True
    >>> transforms += [Transforms.random_grayscale]
    >>> transforms
    [<Transforms.to_tensor: ToTensor()>, <Transforms.fix_channels: FixChannels()>, <Transforms.random_grayscale: RandomGrayscale(p=0.1)>]
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ComposeBase.__init__(self, transforms=self)

    def shape_change(self, input_shape: Union[Tuple[int, ...], torch.Size]) -> Tuple[int, ...]:
        logger.debug(f"shape_change on Compose: input shape: {input_shape}")
        # TODO: Give the impact of this transform on a given input shape.
        for transform in self:
            logger.debug(f"Shape before transform {transform}: {input_shape}")
            if hasattr(transform, "shape_change") and callable(transform.shape_change):
                input_shape = transform.shape_change(input_shape)
            else:
                logger.debug(
                    f"Unable to detect the change of shape caused by "
                    f"transform {transform}, assuming its output has same "
                    f"shape as its input."
                )
        logger.debug(f"Final shape: {input_shape}")
        return input_shape






@encode.register
def encode_transforms(v: Transforms) -> str:
    return v.name

def decode_transforms(v: str) -> Transforms:
    print(f"Decoding a Transforms object: {v}, {Transforms[v]}")
    return Transforms[v]


if __name__ == "__main__":
    import doctest
    doctest.testmod()
