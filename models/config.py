from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, List, Tuple, TypeVar

import torch
from simple_parsing import field
from simple_parsing.utils import JsonSerializable
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from utils import cuda_available, gpus_available

@dataclass
class Config:
    """Settings related to the training setup. """
    log_dir: str = "results" # Logging directory.
    log_interval: int = 10   # How many batches to wait between logging calls.
    seed: int = 1            # Random seed.
    num_classes: int = 10    # Number of output classes in the classifier.
    # Wether or not to use CUDA. Defaults to True when available.
    use_cuda: bool = cuda_available
    
    # Which specific device to use.
    # NOTE: Can be set directly with the command-line! (ex: "--device cuda")
    device: torch.device = torch.device("cuda" if cuda_available else "cpu")
    
    # Wether to train in the IID or Non-Stationary setting.
    non_iid: bool = False
    iid: bool = field(default=None, init=False)

    def __post_init__(self):
        # set the manual seed (for reproducibility)
        torch.manual_seed(self.seed)
        self.iid = not self.non_iid
        
        if self.use_cuda and not cuda_available:
            print("Cannot use the passed value of argument 'use_cuda', as CUDA "
                  "is not available!")
            self.use_cuda = False
        
        if not self.use_cuda:
            self.device = torch.device("cpu")

    def get_dataloaders(self, batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
        """Create the train and test dataloaders using the passed arguments.
                
        Returns
        -------
        Tuple[DataLoader, DataLoader]
            The training and validation dataloaders.
        """

        # TODO: Extract the 'MNIST' logic of this class into a new subclass.
        train_dataset = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
        
        if self.non_iid:
            train_dataset.targets, sort_indices = torch.sort(train_dataset.targets)
            train_dataset.data = train_dataset.data[sort_indices]

        valid_dataset = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())

        if self.non_iid:
            valid_dataset.targets, sort_indices = torch.sort(valid_dataset.targets)
            valid_dataset.data = valid_dataset.data[sort_indices]

        train_loader =  DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=not self.non_iid,
            num_workers=1,
            pin_memory=self.use_cuda,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=self.iid,
            num_workers=1,
            pin_memory=self.use_cuda,
        )
        return train_loader, valid_loader