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

from config import Config
from utils import cuda_available, gpus_available

from .bases import Dataset


@dataclass
class Mnist(Dataset):
    name: str = "MNIST"
    data_dir: str = "../data"

    x_shape: Tuple[int, int, int] = (1, 28, 28)
    y_shape: Tuple[int] = (10,)
    
    non_iid: bool = False
    iid: bool = field(default=None)

    def __post_init__(self):
        self.iid = not self.non_iid

    def get_dataloaders(self, batch_size: int = 64, config: Config=None) -> Tuple[DataLoader, DataLoader]:
        """Create the train and test dataloaders using the passed arguments.
                
        Returns
        -------
        Tuple[DataLoader, DataLoader]
            The training and validation dataloaders.
        """
        config = config or Config()
        train_dataset = datasets.MNIST(self.data_dir, train=True, download=True, transform=transforms.ToTensor())
        valid_dataset = datasets.MNIST(self.data_dir, train=False, download=True, transform=transforms.ToTensor())
        
        if self.non_iid:
            train_dataset.targets, sort_indices = torch.sort(train_dataset.targets)
            train_dataset.data = train_dataset.data[sort_indices]
            valid_dataset.targets, sort_indices = torch.sort(valid_dataset.targets)
            valid_dataset.data = valid_dataset.data[sort_indices]

        train_loader =  DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=not self.non_iid,
            num_workers=1,
            pin_memory=config.use_cuda,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=self.iid,
            num_workers=1,
            pin_memory=config.use_cuda,
        )
        return train_loader, valid_loader
