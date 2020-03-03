import random
from abc import ABC, abstractmethod
from dataclasses import InitVar, dataclass
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
from utils.utils import n_consecutive, to_list


@dataclass
class TaskConfig:
    id: int
    classes: List[int]
    class_counts: List[int]
    start_index: int
    end_index: int = field(default=None, init=False)
    indices: slice = field(init=False, repr=False)
    def __post_init__(self):
        self.classes = to_list(self.classes)
        self.class_counts = to_list(self.class_counts)
        self.end_index = self.start_index + sum(self.class_counts)
        self.indices = slice(self.start_index, self.end_index)


@dataclass  # type: ignore 
class Dataset:
    """
    Represents all the command-line arguments as well as logic related to a Dataset.
    """
    name: str = "default"
    data_dir: str = "../data"
    config: Config = field(default_factory=Config, init=False)
    
    x_shape: ClassVar[Tuple[int, int, int]] = (1, 28, 28)
    y_shape: ClassVar[Tuple[int]] = (10,)
    
    @abstractmethod
    def get_dataloaders(self, config: Config, batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
        """Create the train and test dataloaders using the passed arguments.
                
        Returns
        -------
        Tuple[DataLoader, DataLoader]
            The training and validation dataloaders.
        """
        pass

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
            selected = torch.cat([(old_y==task_class).unsqueeze(0) for task_class in task_classes], dim=0)
            selected_mask = selected.sum(dim=0, dtype=torch.bool)            
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
            
            assert to_list(y.unique()) == to_list(task_classes), y.unique()
            assert x.shape[0] == y.shape[0] == total_count
            
            new_x[current_index: current_index + total_count] = x
            new_y[current_index: current_index + total_count] = y

            current_index += total_count

        dataset.data = new_x
        dataset.targets = new_y
        
        return task_configs
