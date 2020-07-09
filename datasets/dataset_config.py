from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from pl_bolts.datamodules import (CIFAR10DataModule, FashionMNISTDataModule,
                                  ImagenetDataModule, LightningDataModule,
                                  MNISTDataModule, SSLImagenetDataModule)
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import (CIFAR10, CIFAR100, MNIST, FashionMNIST,
                                  ImageNet, VisionDataset)
from torchvision.transforms import Compose, Resize, ToTensor

from simple_parsing import choice, field, list_field
from utils.json_utils import Serializable

from .data_utils import FixChannels, keep_in_memory


@dataclass
class DatasetConfig(Serializable):
    """
    Represents all the configuration options related to a Dataset.
    """
    # which dataset class to use.
    dataset_class: Type[LightningDataModule] = field(default=MNISTDataModule, encoding_fn=str) 
    x_shape: Tuple[int, int, int] = (3, 28, 28)
    num_classes: int = 10

    # Transforms to apply to the data
    transforms: Optional[Callable] = Compose([
        ToTensor(),
        FixChannels(),
    ])
    train_transforms: Optional[Callable] = None
    valid_transforms: Optional[Callable] = None
    test_transforms: Optional[Callable] = None

    target_transforms: Optional[Callable] = None
    # Wether we want to load the dataset to memory.
    keep_in_memory: bool = False

    def __post_init__(self):
        if self.train_transforms is None:
            self.train_transforms = self.transforms
        if self.valid_transforms is None:
            self.valid_transforms = self.transforms
        if self.test_transforms is None:
            self.test_transforms = self.transforms

    def load(self, data_dir: Union[str,Path]) -> LightningDataModule:
        """ Downloads the corresponding train & test datasets and returns them.
        """
        data_dir = str(data_dir)
        assert issubclass(self.dataset_class, LightningDataModule), "Should be using LightningDataModule instead of VisionDatasets whenever possible."
        return self.dataset_class(
            data_dir,
            train_transforms=self.train_transforms,
            val_transforms=self.valid_transforms,
            test_transforms=self.test_transforms,
        )
