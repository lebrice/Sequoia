from typing import Callable, List, TypeVar, Union, Tuple

import torch
from torch import Tensor
from torchvision.transforms import Compose as ComposeBase
from utils.logging_utils import get_logger

logger = get_logger(__file__)

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
            shape_change_method: Optional[Callable] = getattr(transform, "shape_change", None)
            if shape_change_method and callable(shape_change_method):
                input_shape = transform.shape_change(input_shape)  # type: ignore
            else:
                logger.debug(
                    f"Unable to detect the change of shape caused by "
                    f"transform {transform}, assuming its output has same "
                    f"shape as its input."
                )
        logger.debug(f"Final shape: {input_shape}")
        return input_shape
