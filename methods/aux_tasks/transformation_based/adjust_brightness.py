from typing import Any, Callable, List

import torch
import torchvision
from torch import Tensor, nn
from torchvision import transforms as T
from torchvision.transforms import functional as TF

from ..auxiliary_task import AuxiliaryTask
from .bases import RegressTransformationTask, wrap_pil_transform


class AdjustBrightnessTask(RegressTransformationTask):
    """Task that adjusts the brightness of the image.
    
    TODO: Actually read the paper, they use a fancy brightness format that is
    supposedly more akin to the human visual range, etc.
    """
    def __init__(self,
                 brightness_values: List[float]=None,
                 min_brightness: float=0.1,
                 max_brightness: float=2.0,
                 n_calls: int=2,
                 name: str="adjust_brightness",
                 options: RegressTransformationTask.Options=None):
        super().__init__(
            function=wrap_pil_transform(TF.adjust_brightness),
            function_args=brightness_values,
            function_arg_range=(min_brightness, max_brightness),
            n_calls=n_calls,
            name=name,
            options=options or RegressTransformationTask.Options()
        )
