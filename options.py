from dataclasses import dataclass
from typing import Tuple

import torch
from simple_parsing import field
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from utils import cuda_available, gpus_available


@dataclass
class Options:
    """ Set of options for the VAE MNIST Example. """
    batch_size: int = 128   # Input batch size for training.
    epochs: int = 10        # Number of epochs to train.
    seed: int = 1           # Random seed.
    log_interval: int = 10  # How many batches to wait before logging training status.
    
    # Wether to train in the IID or Non-Stationary setting. If passed, the
    # datasets are sorted by labels. If not, they are shuffled.
    non_iid: bool = False
    iid: bool = field(default=None, init=False)
    
    # Wether or not to use CUDA. Defaults to True when available.
    use_cuda: bool = cuda_available

    # Which specific device to use.
    # NOTE: Can be set directly with the command-line! (ex: "--device cuda")
    device: torch.device = torch.device("cuda" if cuda_available else "cpu")

    def __post_init__(self):
        torch.manual_seed(self.seed)
        self.iid = not self.non_iid
        
        if self.use_cuda and not cuda_available:
            print("Cannot use the passed value of argument 'use_cuda', as CUDA is not available!")
            self.use_cuda = False
        
        if not self.use_cuda:
            self.device = torch.device("cpu")

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create the train and test dataloaders using the passed arguments.
                
        Returns
        -------
        Tuple[DataLoader, DataLoader]
            The training and validation dataloaders.
        """

        # TODO: Extract the 'MNIST' logic of this class into a new subclass.
        train_dataset = datasets.MNIST('data', train=True, transform=transforms.ToTensor())
        
        if self.non_iid:
            train_dataset.targets, sort_indices = torch.sort(train_dataset.targets)
            train_dataset.data = train_dataset.data[sort_indices]

        valid_dataset = datasets.MNIST('data', train=False, transform=transforms.ToTensor())

        train_loader =  DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=not self.non_iid,
            num_workers=1,
            pin_memory=self.use_cuda,
        )
        test_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=self.iid,
            num_workers=1,
            pin_memory=self.use_cuda,
        )
        return train_loader, test_loader
