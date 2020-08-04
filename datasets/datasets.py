from enum import Enum
from dataclasses import dataclass
from utils.json_utils import Serializable

from torchvision.datasets import (CIFAR10, CIFAR100, MNIST, FashionMNIST,
                                  ImageNet)
from torchvision.transforms import Compose, Resize, ToTensor, Lambda, Normalize, ToPILImage
import numpy as np
from .dataset_config import DatasetConfig
from .imagenet import ImageNetConfig, ImageNetConfig_Folder

from simple_parsing import choice, field, list_field
from typing import Callable

from pl_bolts.models.self_supervised.simclr import SimCLREvalDataTransform, SimCLRTrainDataTransform


@dataclass
class DatasetsHParams(Serializable):
    """ Options for dataset """
    #train dataset transforms
    transforms: Callable = choice({ 
        'simclr': SimCLRTrainDataTransform(32),
        'None': ToTensor()}, default=ToTensor())
    #test dataset transforms
    transforms_test: Callable = choice({
        'simclr': SimCLREvalDataTransform(32),
        'None': ToTensor()}, default=ToTensor())
    #x_shape
    x_shape: tuple = (3, 32, 32)
    #number of classes to sample
    num_classes:int = 100

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
        x_shape=(3,32,32),
        num_classes=100,
        # transforms = ToTensor(),
        # transforms_test = ToTensor(), 
    )
    cifar100_normalized = DatasetConfig(
        CIFAR100,
        x_shape=(3,32,32),
        num_classes=100,
        transforms = Compose([ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]),
        transforms_test = Compose([ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    )
    
    cifar100_simclrtrnsform = DatasetConfig(
        CIFAR100,
        x_shape=(3,32,32),
        num_classes=100,
        transforms = SimCLRTrainDataTransform(32),
        transforms_test = ToTensor(), #SimCLREvalDataTransform(32)
    )


    imagenet = ImageNetConfig(
        x_shape=(3, 224, 224),
    )
    mini_imagenet = ImageNetConfig_Folder(
        x_shape=(3, 32, 32),
        num_classes=200,
    )