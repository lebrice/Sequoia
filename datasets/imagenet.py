import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from socket import gethostname
from typing import Callable, ClassVar, Dict, Tuple, Type

from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision.datasets import ImageNet, VisionDataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from simple_parsing import field
from .dataset_config import DatasetConfig
from pl_bolts.datamodules import (ImagenetDataModule, LightningDataModule)
from .data_utils import get_imagenet_location

@dataclass
class ImageNetConfig(DatasetConfig):
    dataset_class: Type[VisionDataset] = field(default=ImageNet, encoding_fn=str) 
    transforms: Callable = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace=True),
    ])

    def load(self, data_dir: Path=None, *args, **kwargs) -> ImagenetDataModule:
        """ Downloads the ImageNet dataset.
        """
        if data_dir is not None:
            warnings.warn(UserWarning(
                f"Using data_dir has no effect when using the ImageNet dataset."
            ))
        data_dir = get_imagenet_location()
        return super().load(data_dir=data_dir, *args, **kwargs)
