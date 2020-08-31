""" Transforms and such. Trying to make it possible to parse such from the
command-line. """

from dataclasses import dataclass
from torchvision.transforms import ToTensor, Compose as ComposeBase
from enum import Enum
from torch import Tensor
from typing import List, Callable, Any

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


@dataclass
class FixChannels(Callable[[Tensor], Tensor]):
    """ Transform that fixes the number of channels in input images. 
    
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

from torchvision.transforms import RandomGrayscale


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


from utils.json_utils import encode, register_decoding_fn


@encode.register
def encode_transforms(v: Transforms) -> str:
    return v.name

def decode_transforms(v: str) -> Transforms:
    print(f"Decoding a Transforms object: {v}, {Transforms[v]}")
    return Transforms[v]
register_decoding_fn



if __name__ == "__main__":
    import doctest
    doctest.testmod()