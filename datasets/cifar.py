from dataclasses import dataclass
from typing import Tuple, Type

import torchvision.datasets as v_datasets
import torchvision.transforms as T
from torchvision.datasets import VisionDataset

from config import Config
from datasets.dataset import DatasetConfig


@dataclass
class Cifar10(DatasetConfig):
    name: str = "Cifar10"
    x_shape: Tuple[int, int, int] = (3, 32, 32)
    y_shape: Tuple[int] = (10,)
    dataset_class: Type[VisionDataset] = v_datasets.CIFAR10


@dataclass
class Cifar100(DatasetConfig):
    name: str = "Cifar10"
    x_shape: Tuple[int, int, int] = (3, 32, 32)
    y_shape: Tuple[int] = (100,)
    dataset_class: Type[VisionDataset] = v_datasets.CIFAR100
