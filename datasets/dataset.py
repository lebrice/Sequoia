import random
from abc import ABC, abstractmethod
from dataclasses import InitVar, dataclass
from typing import *

import torch
from simple_parsing import field, choice
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from torchvision import datasets as v_datasets
from torchvision.datasets import VisionDataset
from config import Config
from utils import cuda_available, gpus_available
from utils.utils import n_consecutive, to_list


@dataclass
class TaskConfig:
    classes: List[int]
    class_counts: List[int]
    start_index: int
    end_index: int = field(default=None, init=False)

    def __post_init__(self):
        self.classes = to_list(self.classes)
        self.class_counts = to_list(self.class_counts)
        self.end_index = self.start_index + sum(self.class_counts)
        self.indices: List[int] = list(range(self.start_index, self.end_index))

@dataclass
class DatasetConfig:
    """
    Represents all logic related to a Dataset.

    NOTE (Fabrice): No command-line arguments are created here. Instead, I
    decided to have the arguments related to the dataset be in `config`.
    """
    name: str = "default"

    # which dataset class to use. (TODO: add more of them.)
    dataset_class: ClassVar[Type[v_datasets.VisionDataset]] = field(default=v_datasets.MNIST, repr=False)

    config: Config = field(default_factory=Config)
    
    x_shape: Tuple[int, int, int] = (1, 28, 28)
    y_shape: Tuple[int] = (10,)
    
    train: Optional[VisionDataset] = None
    valid: Optional[VisionDataset] = None

    # The indices where there is a transition between tasks.
    train_tasks: List[TaskConfig] = field(default_factory=list)
    valid_tasks: List[TaskConfig] = field(default_factory=list)

    current_train_task: Optional[TaskConfig] = None
    current_valid_task: Optional[TaskConfig] = None

    transforms: object = transforms.ToTensor()

    _loaded: bool = False

    def load(self, config: Config) -> None:
        if self._loaded:
            return
        self.train = self.dataset_class(config.data_dir, train=True,  download=True, transform=self.transforms)
        self.valid = self.dataset_class(config.data_dir, train=False, download=True, transform=self.transforms)
        self._loaded = True     

    def get_dataloaders(self, config: Config, batch_size: int = 64) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
        """Create the train and test dataloaders using the passed arguments.
                
        You might want to override/extend this method in your subclass.

        Returns
        -------
        Tuple[DataLoader, DataLoader]
            The training and validation dataloaders.
        """
        self.load(config)

        train_loader = None
        if self.train:
            class_incremental = bool(self.train_tasks)

            train_dataset = self.train
            if self.current_train_task:
                indices = list(range(self.current_train_task.start_index, self.current_train_task.end_index))
                train_dataset = torch.utils.data.Subset(self.train, indices)

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=not class_incremental,
                num_workers=1,
                pin_memory=config.use_cuda,
            )

        valid_loader = None
        if self.valid:
            class_incremental = bool(self.valid_tasks)

            valid_dataset = self.valid
            if self.current_valid_task:
                indices = list(range(self.current_valid_task.start_index, self.current_valid_task.end_index))
                valid_dataset = torch.utils.data.Subset(self.valid, indices)

            valid_loader = DataLoader(
                valid_dataset,
                batch_size=batch_size,
                shuffle=not class_incremental,
                num_workers=1,
                pin_memory=config.use_cuda,
            )
        return train_loader, valid_loader


def make_class_incremental(dataset: VisionDataset,
                           n_classes_per_task: int=2,
                           random_ordering: bool=False) -> List[TaskConfig]:
    """Rearranges the given dataset in-place, making it class-incremental.

    By default, the classes will be presented in order and in pairs of two.

    Parameters
    ----------
    - dataset : VisionDataset
    
        A `torchvision.datasets.VisionDataset` instance to modify in-place.
    - n_classes_per_task : int, optional, by default 2
    
        Number of classes per task. The data within a task is shuffled.
    - random_ordering : bool, optional, by default False
    
        Wether the ordering of classes should itself be random.
    
    Returns
    -------
    List[TaskConfig]
        A list of `TaskConfig` objects that give useful information about the
        resulting tasks (classes, task_boundaries, etc.)
    """
    task_configs: List[TaskConfig] = []

    dataset.targets, train_sort_indices = torch.sort(dataset.targets)
    dataset.data = dataset.data[train_sort_indices]
    classes, counts = dataset.targets.unique_consecutive(return_counts=True)
    classes_and_counts = list(zip(classes, counts))
    
    if random_ordering:
        random.shuffle(classes_and_counts)
    n = n_classes_per_task

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
