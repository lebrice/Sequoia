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


class Transforms(Enum):
    """ Enum of possible transforms. 

    By having this as an Enum, we can choose which transforms to use from the
    command-line.
    This also makes it easier to check for identity, e.g. to check wether a
    particular transform was used.  

    TODO: Add the SimCLR/MOCO/etc transforms from  https://pytorch-lightning-bolts.readthedocs.io/en/latest/transforms.html
    """
    fix_channels = FixChannels()
    to_tensor = ToTensorIfNeeded()

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
    """ Extend the Compose class of torchvision with the list methods. """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ComposeBase.__init__(self, transforms=self)

