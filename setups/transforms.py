""" Transforms and such. Trying to make it possible to parse such from the
command-line. """


from torchvision.transforms import ToTensor, Compose as ComposeBase
from enum import Enum
from torch import Tensor
from typing import List, Callable, Any
from datasets.data_utils import FixChannels


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

