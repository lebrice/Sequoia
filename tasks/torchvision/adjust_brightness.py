import torch
import torchvision
from torch import nn, Tensor
import torchvision
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from typing import Callable, Any
from tasks.bases import AuxiliaryTask
from .transformation import ClassifyTransformationTask


class AdjustBrightnessTask(ClassifyTransformationTask):
    """Task that adjusts the brightness of the image.
    
    NOTE: right now, it tries to classify which of the arguments was used.
    TODO: Might be a good idea to regress the value instead!
    """
    def __init__(self, options: AuxiliaryTask.Options):
        super().__init__(
            function=self.pil_transform(TF.adjust_brightness),
            function_args=[0.5, 1.5],
            options=options,
        )