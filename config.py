from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, List, Tuple, TypeVar

import torch
from simple_parsing import field
from simple_parsing.helpers import JsonSerializable
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from utils import cuda_available, gpus_available

@dataclass
class Config:
    """Settings related to the training setup. """

    debug: bool = False      # enable debug mode.
    verbose: bool = False    # enable verbose mode.

    log_dir: str = "results" # Logging directory.
    log_interval: int = 10   # How many batches to wait between logging calls.
    
    random_seed: int = 1            # Random seed.
    use_cuda: bool = cuda_available # Wether or not to use CUDA.
    
    # Which specific device to use.
    # NOTE: Can be set directly with the command-line! (ex: "--device cuda")
    device: torch.device = torch.device("cuda" if cuda_available else "cpu")
    
    wandb: str = ""  # Wandb setting (TODO)

    def __post_init__(self):
        # set the manual seed (for reproducibility)
        torch.manual_seed(self.random_seed)
        
        if self.use_cuda and not cuda_available:
            print("Cannot use the passed value of argument 'use_cuda', as CUDA "
                  "is not available!")
            self.use_cuda = False
        
        if not self.use_cuda:
            self.device = torch.device("cpu")

# shared config object.
## TODO: unused, but might be useful!
config: Config = Config()
