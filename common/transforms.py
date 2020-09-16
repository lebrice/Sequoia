""" Transforms and such. Trying to make it possible to parse such from the
command-line.

Also, playing around with the idea of adding the ability to predict the change
in shape resulting from the transforms, à-la-Tensorflow.

"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, List, Tuple, Union

import torch
from torch import Tensor
from torchvision.transforms import Compose as ComposeBase
from torchvision.transforms import RandomGrayscale, ToTensor
from common.dims import Dims
from utils.json_utils import encode, register_decoding_fn
from utils.logging_utils import get_logger
from pl_bolts.models.self_supervised.simclr import SimCLRTrainDataTransform, SimCLREvalDataTransform
logger = get_logger(__file__)


@dataclass
class ToTensorIfNeeded(ToTensor):

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, Tensor):
            return pic
        return super().__call__(pic)

    def shape_change(self, input_shape: Union[Tuple[int, ...], torch.Size]) -> Tuple[int, ...]:
        return input_shape


@dataclass
class FixChannels(Callable[[Tensor], Tensor]):
    """ Transform that fixes the number of channels in input images to be 3.
    
    For instance, if the input shape is:
    [28, 28] -> [3, 28, 28] (copy the image three times)
    [1, 28, 28] -> [3, 28, 28] (same idea)
    [10, 1, 28, 28] -> [10, 3, 28, 28] (keep batch intact, do the same again.)
    
    """
    def __call__(self, x: Tensor) -> Tensor:
        if x.ndim == 2:
            x = x.reshape([1, *x.shape])
            x = x.repeat(3, 1, 1)
        if x.ndim == 3 and x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        if x.ndim == 4 and x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return x
    
    def shape_change(self, input_shape: Union[Tuple[int, ...], torch.Size]) -> Tuple[int, ...]:
        dims = len(input_shape)
        if dims == 2:
            return (3, *input_shape)
        elif dims == 3 and input_shape[0] == 1:
            return (3, *input_shape[1:])
        elif dims == 4 and input_shape[1] == 1:
            return (input_shape[0], 3, *input_shape[2:])
        return input_shape

@dataclass
class ChannelsFirst(Callable[[Tensor], Tensor]):
    def __call__(self, x: Tensor) -> Tensor:
        if x.ndimension() == 3:
            return x.permute(2, 0, 1)
        if x.ndimension() == 4:
            return x.permute(0, 3, 1, 2)
        return x

    def shape_change(self, input_shape: Union[Tuple[int, ...], torch.Size]) -> Tuple[int, ...]:
        ndim = len(input_shape)
        if ndim == 3:
            return tuple(input_shape[i] for i in (2, 0, 1))
        elif ndim == 4:
            return tuple(input_shape[i] for i in (0, 3, 1, 2))         
        return input_shape

@dataclass
class ChannelsFirstIfNeeded(ChannelsFirst):
    def __call__(self, x: Tensor) -> Tensor:
        if x.shape[-1] in {1, 3}:
            return super().__call__(x)
        return x

    def shape_change(self, input_shape: Union[Tuple[int, ...], torch.Size]) -> Tuple[int, ...]:
        if input_shape[-1] in {1, 3}:
            return super().shape_change(input_shape)
        else:
            return input_shape


@dataclass
class ChannelsLast(Callable[[Tensor], Tensor]):
    def __call__(self, x: Tensor) -> Tensor:
        if x.ndimension() == 3:
            return x.permute(1, 2, 0)
        if x.ndimension() == 4:
            return x.permute(0, 2, 3, 1)
        return x
    
    def shape_change(self, input_shape: Union[Tuple[int, ...], torch.Size]) -> Tuple[int, ...]:
        ndim = len(input_shape)
        if ndim == 3:
            new_shape = tuple(input_shape[i] for i in (1, 2, 0))
        elif ndim == 4:
            new_shape = tuple(input_shape[i] for i in (0, 2, 3, 1))
        return new_shape

@dataclass
class ChannelsLastIfNeeded(ChannelsLast):
    def __call__(self, x: Tensor) -> Tensor:
        if x.ndimension() == 4 and x.shape[1] in {1, 3}:
            return super().__call__(x)
        if x.ndimension() == 3 and x.shape[0] in {1, 3}:
            return super().__call__(x)
        return x
    
    def shape_change(self, input_shape: Union[Tuple[int, ...], torch.Size]) -> Tuple[int, ...]:
        ndims = len(input_shape)
        if ndims == 4 and input_shape[1] in {1, 3}:
            return super().shape_change(input_shape)
        if ndims == 3 and input_shape[0] in {1, 3}:
            return super().shape_change(input_shape)
        return input_shape

class Transforms(Enum):
    """ Enum of possible transforms. 

    By having this as an Enum, we can choose which transforms to use from the
    command-line.
    This also makes it easier to check for identity, e.g. to check wether a
    particular transform was used.  

    TODO: Add the SimCLR/MOCO/etc transforms from  https://pytorch-lightning-bolts.readthedocs.io/en/latest/transforms.html
    TODO: Figure out a way to let people customize the arguments to the transforms?
    """
    fix_channels = FixChannels()
    to_tensor = ToTensorIfNeeded()
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

# TODO: Add the SimCLR transforms.
# class SimCLRTrainTransform(SimCLRTrainDataTransform):
#     def __call

class Compose(List[Transforms], ComposeBase):
    """ Extend the Compose class of torchvision with methods of `list`.
    
    This can also be passed in members of the `Transforms` enum, which makes it
    possible to do something like this:
    >>> from transforms import Compose, Transforms
    >>> transforms = Compose([Transforms.to_tensor, Transforms.fix_channels,])
    >>> Transforms.fix_channels in transforms
    True
    >>> transforms += [Transforms.random_grayscale]
    >>> transforms
    [<Transforms.to_tensor: ToTensorIfNeeded()>, <Transforms.fix_channels: FixChannels()>, <Transforms.random_grayscale: RandomGrayscale(p=0.1)>]
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ComposeBase.__init__(self, transforms=self)

    def shape_change(self, input_shape: Union[Tuple[int, ...], torch.Size]) -> Tuple[int, ...]:
        logger.debug(f"shape_change on Compose: input shape: {input_shape}")
        # TODO: Give the impact of this transform on a given input shape.
        for transform in self:
            logger.debug(f"Shape before transform {transform}: {input_shape}")
            input_shape = transform.shape_change(input_shape)
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
