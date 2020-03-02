from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import *

import torch
from simple_parsing import field
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils import data

from config import Config
from utils import cuda_available, gpus_available
from .bases import Dataset


@dataclass
class Mnist(Dataset):
    name: str = "MNIST"
    data_dir: str = "../data"

    x_shape: Tuple[int, int, int] = (1, 28, 28)
    y_shape: Tuple[int] = (10,)
    
    train: data.Dataset = field(default=None, init=False)
    valid: data.Dataset = field(default=None, init=False)

    class_incremental: bool = False
    n_classes_per_task: int = 1

    def __post_init__(self):
        self.train = datasets.MNIST(self.data_dir, train=True,  download=True, transform=transforms.ToTensor())
        self.valid = datasets.MNIST(self.data_dir, train=False, download=True, transform=transforms.ToTensor())

        if self.class_incremental:
            self.train.targets, train_sort_indices = torch.sort(self.train.targets)
            self.valid.targets, valid_sort_indices = torch.sort(self.valid.targets)
            self.train.data = self.train.data[train_sort_indices]
            self.valid.data = self.valid.data[valid_sort_indices]
            self.classes = self.train.targets.unique()
            print(self.classes)

    def get_dataloaders(self, batch_size: int = 64, config: Config=None) -> Tuple[DataLoader, DataLoader]:
        """Create the train and test dataloaders using the passed arguments.
                
        Returns
        -------
        Tuple[DataLoader, DataLoader]
            The training and validation dataloaders.
        """
        config = config or Config()
        
        train_loader =  DataLoader(
            self.train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
            pin_memory=config.use_cuda,
        )
        valid_loader = DataLoader(
            self.valid,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
            pin_memory=config.use_cuda,
        )
        return train_loader, valid_loader
