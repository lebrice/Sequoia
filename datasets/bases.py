from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import *

import torch
from simple_parsing import field
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from utils import cuda_available, gpus_available
from config import Config

@dataclass  # type: ignore 
class Dataset:
    """
    Represents all the command-line arguments as well as logic related to a Dataset.
    """
    name: str = "default"
    data_dir: str = "../data"
    
    @abstractmethod
    def get_dataloaders(self, config: Config, batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
        """Create the train and test dataloaders using the passed arguments.
                
        Returns
        -------
        Tuple[DataLoader, DataLoader]
            The training and validation dataloaders.
        """
        pass
