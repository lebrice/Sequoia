from dataclasses import dataclass
from typing import Tuple, Type, ClassVar

import torchvision.datasets as v_datasets
import torchvision.transforms as T
from torchvision.datasets import FashionMNIST, VisionDataset

from datasets.dataset import DatasetConfig

@dataclass
class Mnist(DatasetConfig):
    name: str = "MNIST"
    x_shape: Tuple[int, int, int] = (1, 28, 28)
    y_shape: Tuple[int] = (10,)
    dataset_class: ClassVar[Type[VisionDataset]] = v_datasets.MNIST
    transforms = T.Compose([
        T.ToTensor(),
        lambda x: x.reshape(1, 28, 28)
    ])
