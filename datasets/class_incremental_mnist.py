from dataclasses import dataclass, field
from typing import *

from datasets.mnist import Mnist
import torch
from torch import Tensor

@dataclass
class ClassIncrementalConfig:
    classes_per_task: int = 2
 

@dataclass
class ClassIncrementalMnist(Dataset):
    config: ClassImcrementalConfig = ClassIncrementalConfig()
    datasets: Dict[int, Tuple[Tensor, Tensor]]

    def __post_init__(self):
        super().__post_init__()
        # sort the data per target.
        self.train.targets, train_sort_indices = torch.sort(self.train.targets)
        self.valid.targets, valid_sort_indices = torch.sort(self.valid.targets)
        self.train.data = self.train.data[train_sort_indices]
        self.valid.data = self.valid.data[valid_sort_indices]

        self.datasets = {}





    def get_dataloaders(self, task_index: int, batch_size: int = 64, config: Config=None) -> Tuple[DataLoader, DataLoader]:
        """Create the train and test dataloaders using the passed arguments.
                
        Returns
        -------
        Tuple[DataLoader, DataLoader]
            The training and validation dataloaders.
        """
        # TODO: Do an actual continual learning setting.
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