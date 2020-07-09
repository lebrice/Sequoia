from enum import Enum

from torch import Tensor
from torchvision import transforms
from torchvision.datasets import (CIFAR10, CIFAR100, MNIST, FashionMNIST,
                                  ImageNet)
from torchvision.transforms import Compose, Resize, ToTensor
from pl_bolts.datamodules import (CIFAR10DataModule, FashionMNISTDataModule,
                                  ImagenetDataModule, LightningDataModule,
                                  MNISTDataModule, SSLImagenetDataModule)
from .dataset_config import DatasetConfig
from .imagenet import ImageNetConfig
import torch
from .data_utils import FixChannels


class Datasets(Enum):
    """ Choice of dataset. """
    mnist = DatasetConfig(
        MNISTDataModule,
        x_shape=(3, 28, 28),
        num_classes=10,
        transforms=transforms.Compose([
            ToTensor(),
            FixChannels(),
        ])
    )
    fashion_mnist = DatasetConfig(
        FashionMNISTDataModule,
        x_shape=(3, 28, 28),
        num_classes=10,
        transforms=transforms.Compose([
            ToTensor(),
            FixChannels(),
        ])
    )
    cifar10 = DatasetConfig(
        CIFAR10DataModule,
        x_shape=(3, 32, 32),
        num_classes=10,
    )
    # cifar100 = DatasetConfig(
    #     CIFAR100,
    #     x_shape=(3, 32, 32),
    #     num_classes=100,
    # )
    imagenet = ImageNetConfig()
