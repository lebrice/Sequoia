from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from sequoia.common.layers import Flatten
from sequoia.common.loss import Loss
from sequoia.common.metrics import get_metrics
from ..auxiliary_task import AuxiliaryTask

from .bases import ClassifyTransformationTask, wrap_pil_transform


def rotate(x: Tensor, angle: int) -> Tensor:
    """Rotates the given tensor `x` by an angle `angle`.

    Currently only supports multiples of 90 degrees.
    
    Args:
        x (Tensor): An image or a batch of images, with shape [(b), C, H, W]
        angle (int): An angle. Currently only supports {0, 90, 180, 270}.
    
    Returns:
        Tensor: The tensor x, rotated by `angle` degrees counter-clockwise.
    
    Example:
    >>> import torch
    >>> x = torch.Tensor([
    ...   [1, 2, 3],
    ...   [4, 5, 6],
    ...   [7, 8, 9],
    ... ])
    >>> print(x)
    tensor([[1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 9.]])
    >>> x = x.view(1, 3, 3)
    >>> x_rot = rotate(x, 90)
    >>> print(x_rot.shape)
    torch.Size([1, 3, 3])
    >>> print(x_rot)
    tensor([[[3., 6., 9.],
             [2., 5., 8.],
             [1., 4., 7.]]])
    """
    
    # TODO: Test that this works.
    assert angle % 90 == 0, "can only rotate 0, 90, 180, or 270 degrees for now."
    k = angle // 90
    # BUG: Very rarely, this condition won't work! (More specifically, only on the last batch of data!)
    # assert min(x.shape) == x.shape[-3], f"Image should be in [(b) C H W] format. (image shape: {x.shape}" 
    return x.rot90(k, dims=(-2,-1))


if __name__ == "__main__":
    import doctest
    doctest.testmod()


class RotationTask(ClassifyTransformationTask):
    @dataclass
    class Options(ClassifyTransformationTask.Options):
        """Command-line options for the Transformation-based auxiliary task."""
        # Wether or not both the original and transformed codes should be passed
        # to the auxiliary layer in order to detect the transformation.
        # TODO: Maybe try with this set to False, to learn "innate" orientation rather than relative orientation. 
        compare_with_original: bool = True

    def __init__(self, name="rotation", options: "RotationTask.Options"=None):
        super().__init__(
            function=rotate,
            function_args=[0, 90, 180, 270],
            name=name,
            options=options or RotationTask.Options(),
        )
