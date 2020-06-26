from enum import Enum

from torchvision.datasets import (CIFAR10, CIFAR100, MNIST, FashionMNIST,
                                  ImageNet)
from torchvision.transforms import Compose, Resize, ToTensor

from .dataset_config import DatasetConfig
from .imagenet import ImageNetConfig, ImageNetConfig_Folder

class Datasets(Enum):
    """ Choice of dataset. """
    mnist = DatasetConfig(
        MNIST,
        x_shape=(1, 28, 28),
        num_classes=10,
    )
    fashion_mnist = DatasetConfig(
        FashionMNIST,
        x_shape=(1, 28, 28),
        num_classes=10,
    )
    cifar10 = DatasetConfig(
        CIFAR10,    
        x_shape=(3, 32, 32),
        num_classes=10,
    )
    cifar100 = DatasetConfig(
        CIFAR100,
        x_shape=(3, 32, 32),
        num_classes=100,
    )
    imagenet = ImageNetConfig(
        x_shape=(3, 224, 224),
    )
    mini_imagenet = ImageNetConfig(
        x_shape=(3, 32, 32),
        num_classes=200,
    )