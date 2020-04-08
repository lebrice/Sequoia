from dataclasses import dataclass
from typing import Tuple, Type

import torchvision.datasets as v_datasets
import torchvision.transforms as T
from torchvision.datasets import VisionDataset

from datasets.dataset import DatasetConfig


@dataclass
class FashionMnist(DatasetConfig):
    name: str = "FashionMNIST"
    x_shape: Tuple[int, int, int] = (1, 28, 28)
    y_shape: Tuple[int] = (10,)
    dataset_class: Type[VisionDataset] = v_datasets.FashionMNIST
    