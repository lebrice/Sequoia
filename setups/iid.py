from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import (Any, Callable, ClassVar, Dict, List, Optional, Tuple, Type,
                    Union)

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as transform_lib
from torchvision.datasets import (CIFAR10, CIFAR100, MNIST, FashionMNIST,
                                  ImageNet, VisionDataset)
from torchvision.transforms import Compose, Resize, ToTensor

import pl_bolts
from pl_bolts.datamodules import (CIFAR10DataModule, FashionMNISTDataModule,
                                  ImagenetDataModule, LightningDataModule,
                                  MNISTDataModule, SSLImagenetDataModule)
from pl_bolts.datamodules.mnist_datamodule import MNISTDataModule
from simple_parsing import choice, field, list_field
from utils.json_utils import Serializable, encode, register_decoding_fn

from datasets.data_utils import FixChannels, keep_in_memory, train_valid_split

# data_dir = Path("data")
# data_module = MNISTDataModule(data_dir, val_split=5000, num_workers=16, normalize=False)
# raise NotImplementedError("TODO")
from .base import PassiveEnvironment


class IIDEnvironment(LightningDataModule, PassiveEnvironment):
    def __init__(self)




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

    available_datasets: ClassVar[Dict[str, Type[LightningDataModule]]] = {
        "mnist": MNISTDataModule,
        "fashion_mnist": FashionMNISTDataModule,
        "cifar10": CIFAR10DataModule,
        "imagenet": ImagenetDataModule,
    }
    # Which setup / dataset to use.
    # The setups/dataset are implemented as `LightningDataModule`s. 
    dataset: str = choice(available_datasets.keys(), default="mnist")

    _default_transform: ClassVar[Callable] = Compose([
        ToTensor(),
        FixChannels(),
    ])

    # TODO: Currently trying to find a way to specify the transforms from the command-line.
    transforms: List[Transforms] = field(default=_default_transform, to_dict=False)
    train_transforms: List[Transforms] = list_field(to_dict=False)
    valid_transforms: List[Transforms] = list_field(to_dict=False)
    test_transforms: List[Transforms] = list_field(to_dict=False)

    def __post_init__(self):
        self.transforms = compose(self.transforms)
        self.train_transforms = compose(self.train_transforms) or self.transforms
        self.valid_transforms = compose(self.valid_transforms) or self.transforms
        self.test_transforms = compose(self.test_transforms) or self.transforms

    def load(self, data_dir: Path, valid_fraction: float=0.2) -> LightningDataModule:
        """ Downloads the corresponding train, valid and test datasets and returns them.
        """
        dataset_class = type(self).available_datasets[self.dataset]
        return dataset_class(
            data_dir=data_dir,
            train_transforms=self.train_transforms,
            val_transforms=self.valid_transforms,
            test_transforms=self.test_transforms,
        )
