from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, Tuple, Type

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import (CIFAR10, CIFAR100, MNIST, FashionMNIST,
                                  ImageNet, VisionDataset)
from torchvision.transforms import Compose, Resize, ToTensor

from simple_parsing import choice, field
from utils.json_utils import Serializable
from .data_utils import keep_in_memory
from simple_parsing import MutableField as mutable_field
from pl_bolts.models.self_supervised.simclr import SimCLREvalDataTransform, SimCLRTrainDataTransform

@dataclass
class DatasetConfig(Serializable):
    """
    Represents all the configuration options related to a Dataset.
    """
    
    # which dataset class to use.
    dataset_class: Type[VisionDataset] = field(default=MNIST, encoding_fn=str) 
    x_shape: Tuple[int, int, int] = (1, 28, 28)
    num_classes: int = 10
    # Transforms to apply to the data
    transforms: Optional[Callable] = ToTensor()
    transforms_test: Optional[Callable] = ToTensor()
    target_transforms: Optional[Callable] = None
    # Wether we want to load the dataset to memory.
    keep_in_memory: bool = True

    
    @property
    def name(self):
        return self.dataset_class.__name__
    def load(self, data_dir: Path, download: bool=True, train_transform=None, valid_transform=None, test_transform=None) -> Tuple[Dataset, Dataset]:
        """ Downloads the corresponding train & test datasets and returns them.
        """
        # Use the data_dir argument if given, otherwise use "./data"
        # train_transform = train_transform if train_transform is not None else self.transforms
        # test_transform = test_transform if test_transform is not None else self.transforms_test
        # valid_transform = valid_transform if valid_transform is not None else self.transforms_test
        
        train = self.dataset_class(data_dir, train=True, download=download, transform=train_transform)
        valid = self.dataset_class(data_dir, train=True, download=download, transform=valid_transform)
        test  = self.dataset_class(data_dir, train=False, download=download, transform=test_transform)
        if self.keep_in_memory:
            keep_in_memory(train)
            keep_in_memory(test)
        return train, valid, test


class DatasetConfig_simclr_augment(DatasetConfig):
    def load(self, data_dir: Path, download: bool=True) -> Tuple[Dataset, Dataset]:
        """ Downloads the corresponding train & test datasets and returns them.
        """
        
        # Use the data_dir argument if given, otherwise use "./data"
        train = self.dataset_class(data_dir, train=True, download=download, transform=self.transforms)
        test  = self.dataset_class(data_dir, train=False, download=download, transform=self.transforms_test)
        if self.keep_in_memory:
            keep_in_memory(train)
            keep_in_memory(test)
        return train, test
