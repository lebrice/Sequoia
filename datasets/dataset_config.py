from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, ClassVar, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as transform_lib
from torchvision.datasets import (CIFAR10, CIFAR100, MNIST, FashionMNIST,
                                  ImageNet, VisionDataset)
from torchvision.transforms import Compose, Resize, ToTensor

from pl_bolts.datamodules import (CIFAR10DataModule, FashionMNISTDataModule,
                                  ImagenetDataModule, LightningDataModule,
                                  MNISTDataModule, SSLImagenetDataModule)
from simple_parsing import choice, field, list_field
from utils.json_utils import Serializable

from .data_utils import FixChannels, keep_in_memory, train_valid_split


class Transforms(Enum):
    """ Enum of possible transforms. 
    TODO: Maybe use this to create a customizable input pipeline (with the Simclr MoCo/etc augments?)
    """
    fix_channels = FixChannels()
    to_tensor = transform_lib.ToTensor()

    def __mult__(self, other: "Transforms"):
        # TODO: maybe use multiplication as composition?
        return NotImplemented
    @classmethod
    def _missing_(cls, value: Any):
        return cls[value]
        

def compose(transforms) -> Compose:
    if isinstance(transforms, (list, tuple)):
        if len(transforms) == 1:
            return transforms[0]
        elif len(transforms) > 1:
            return Compose(transforms)
    return transforms


@dataclass
class DatasetConfig(Serializable):
    """
    Represents all the configuration options related to a Dataset.
    """
    dataset: Type[LightningDataModule] = choice({
        "mnist": MNISTDataModule,
        "fashion_mnist": FashionMNISTDataModule,
        "cifar10": CIFAR10DataModule,
        "imagenet": ImagenetDataModule,
    }, default=MNISTDataModule)

    _default_transform: ClassVar[Callable] = Compose([
        ToTensor(),
        FixChannels(),
    ])

    # TODO: Currently trying to find a way to specify the transforms from the command-line.
    transforms: List[Transforms] = field(default=_default_transform)
    train_transforms: List[Transforms] = list_field()
    valid_transforms: List[Transforms] = list_field()
    test_transforms: List[Transforms] = list_field()

    def __post_init__(self):
        self.transforms = compose(self.transforms)
        self.train_transforms = compose(self.train_transforms) or self.transforms
        self.valid_transforms = compose(self.valid_transforms) or self.transforms
        self.test_transforms = compose(self.test_transforms) or self.transforms

    def load(self, data_dir: Path, valid_fraction: float=0.2) -> LightningDataModule:
        """ Downloads the corresponding train, valid and test datasets and returns them.
        """
        return self.dataset(
            data_dir=data_dir,
            train_transforms=self.train_transforms,
            val_transforms=self.valid_transforms,
            test_transforms=self.test_transforms,
        )


        # # Use the data_dir argument if given, otherwise use "./data"
        # train = self.dataset_class(data_dir, train=True,  download=download, transform=self.transforms)
        # test  = self.dataset_class(data_dir, train=False, download=download, transform=self.transforms)
        # valid: Optional[Dataset] = None
        
        # if valid_fraction > 0:
        #     train, valid = train_valid_split(train, valid_fraction)
        
        # if self.keep_in_memory:
        #     keep_in_memory(train)
        #     keep_in_memory(test)
        #     if valid is not None:
        #         keep_in_memory(valid)

        # return train, valid, test
