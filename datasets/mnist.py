import os
import random
from abc import ABC
from dataclasses import InitVar, dataclass
from pathlib import Path
from typing import *

import torch
from simple_parsing import choice, field
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as T
from torchvision.utils import save_image

from config import Config
from utils import cuda_available, gpus_available
from utils.utils import n_consecutive

from .bases import Dataset, TaskConfig

@dataclass
class Mnist(Dataset):
    name: str = "MNIST"

    def __post_init__(self):
        self.x_shape: Tuple[int, int, int] = (1, 28, 28)
        self.y_shape: Tuple[int] = (10,)
        self.transforms = T.Compose([
            T.ToTensor(),
            lambda x: x.reshape(1, 28, 28)
        ])
        self.train: Optional[datasets.MNIST] = None
        self.valid: Optional[datasets.MNIST] = None
        self.config: Config = None

        # The indices where there is a transition between tasks.
        self.train_tasks: List[TaskConfig] = []
        self.valid_tasks: List[TaskConfig] = []

    def load(self, config: Config) -> None:
        self.train = datasets.MNIST(self.data_dir, train=True,  download=True, transform=self.transforms)
        self.valid = datasets.MNIST(self.data_dir, train=False, download=True, transform=self.transforms)
        
        if config.class_incremental:
            self.train_tasks = self.make_class_incremental(self.train)
            self.valid_tasks = self.make_class_incremental(self.valid)
            if config.debug:
                print("Class Incremental Setup:")
                print("Training tasks:")
                print(*self.train_tasks, sep="\n")
                print("Validation tasks:")
                print(*self.valid_tasks, sep="\n")
            
            self.save_images_for_each_task(
                self.train,
                self.train_tasks,
                folder_name="train_tasks"
            )
            self.save_images_for_each_task(
                self.valid,
                self.valid_tasks,
                folder_name="valid_tasks"
            )
     
    def get_dataloaders(self, config: Config, batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
        """Create the train and test dataloaders using the passed arguments.
                
        Returns
        -------
        Tuple[DataLoader, DataLoader]
            The training and validation dataloaders.
        """
        self.config = config

        if self.train is None or self.valid is None:
            self.load(config)

        assert self.train is not None 
        train_loader =  DataLoader(
            self.train,
            batch_size=batch_size,
            shuffle=not config.class_incremental,
            num_workers=1,
            pin_memory=config.use_cuda,
        )
        assert self.valid is not None
        valid_loader = DataLoader(
            self.valid,
            batch_size=batch_size,
            shuffle=not config.class_incremental,
            num_workers=1,
            pin_memory=config.use_cuda,
        )
        return train_loader, valid_loader

    def save_images_for_each_task(self,
                                  dataset: datasets.MNIST,
                                  tasks: List[TaskConfig],
                                  folder_name:str="tasks"):
        os.makedirs(os.path.join(self.config.log_dir, folder_name), exist_ok=True)
        n = 64
        for task in tasks:
            start = task.start_index
            stop = start + n 
            sample = dataset.data[start: start+n].view(n, 1, 28, 28).float()
            save_image(sample, os.path.join(self.config.log_dir, folder_name, f"task_data_{task.id}.png"))
            