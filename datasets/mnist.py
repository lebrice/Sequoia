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

from .bases import Dataset


@dataclass
class TaskConfig:
    id: int
    classes: List[int]
    class_counts: List[int]
    start_index: int
    end_index: int = field(init=False)

    def __post_init__(self):
        self.end_index = self.start_index + sum(self.class_counts)


@dataclass
class Mnist(Dataset):
    name: str = "MNIST"

    def __post_init__(self):
        self.x_shape: Tuple[int, int, int] = (1, 28, 28)
        self.y_shape: Tuple[int] = (10,)
        self.transforms = T.Compose([
            T.ToTensor(),
            lambda x: x.reshape([1, 28, 28])
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

    def make_class_incremental(self, dataset: datasets.MNIST) -> List[TaskConfig]:
        task_configs: List[TaskConfig] = []

        dataset.targets, train_sort_indices = torch.sort(dataset.targets)
        dataset.data = dataset.data[train_sort_indices]
        classes, counts = dataset.targets.unique_consecutive(return_counts=True)
        classes_and_counts = list(zip(classes, counts))
        
        if self.config.random_class_ordering:
            random.shuffle(classes_and_counts)
        n = self.config.n_classes_per_task
    
        old_x = dataset.data
        old_y = dataset.targets
        new_x = torch.empty_like(dataset.data)
        new_y = torch.empty_like(dataset.targets)
        
        current_index = 0
        for i, task_classes_and_counts in enumerate(n_consecutive(classes_and_counts, n)):
            task_classes, task_counts = zip(*task_classes_and_counts)
            selected_mask = sum([old_y==task_class for task_class in task_classes])
            total_count = sum(task_counts).item()  # type: ignore
            
            task_config = TaskConfig(
                id=i,
                classes=list(task_classes),
                class_counts=list(task_counts),
                start_index= current_index,
            )
            task_configs.append(task_config)

            permutation = torch.randperm(total_count)

            x = old_x[selected_mask]
            y = old_y[selected_mask]
            x = x[permutation]
            y = y[permutation]
            assert x.shape[0] == y.shape[0] == total_count
            
            new_x[current_index: current_index + total_count] = x
            new_y[current_index: current_index + total_count] = y

            current_index += total_count
            
        dataset.data = new_x
        dataset.targets = new_y
        
        return task_configs
     
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
            