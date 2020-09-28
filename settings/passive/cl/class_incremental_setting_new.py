from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Type

import pytorch_lightning as pl
import torch
from continuum import ClassIncremental, split_train_val
from continuum.datasets import *
from continuum.datasets import _ContinuumDataset
from pytorch_lightning import LightningDataModule, Trainer
from settings.base import Setting
from simple_parsing import choice
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ChainDataset, ConcatDataset, Dataset
from utils.logging_utils import get_logger

logger = get_logger(__file__)

@dataclass
class ClassIncrementalSetting(Setting):
    # Class variable holding all the available datasets.
    available_datasets: ClassVar[Dict[str, Type[Dataset]]] = {
        c.__name__.lower(): c
        for c in [
            CORe50, CORe50v2_79, CORe50v2_196, CORe50v2_391,
            CIFARFellowship, Fellowship, MNISTFellowship,
            ImageNet100, ImageNet1000,
            MultiNLI,
            CIFAR10, CIFAR100, EMNIST, KMNIST, MNIST, QMNIST, FashionMNIST,
            PermutedMNIST, RotatedMNIST,
        ]
    }
    data_dir: Path = Path("data")
    # A continual dataset to use. (Should be taken from the continuum package).
    dataset: str = choice(available_datasets.keys(), default="mnist")

    # Number of classes per task.
    n_classes_per_task: int = 2

    def __post_init__(self):
        super().__post_init__()
        self.dataset_class = self.available_datasets[self.dataset]
        self.train_datasets: List[Dataset] = []
        self.val_datasets: List[Dataset] = []
        self.test_datasets: List[Dataset] = []
        self.current_task_id: int = 0

    def prepare_data(self, *args, **kwargs):
        self.dataset_class(self.data_dir, download=True, train=True)
        self.dataset_class(self.data_dir, download=True, train=False)

    def setup(self, stage: Optional[str] = None):
        train = stage == "fit"
        # TODO: Remove this, just debugging:
        assert train, stage
        dataset = self.dataset_class(self.data_dir, download=False, train=train)
        scenario = ClassIncremental(
            dataset,
            increment=self.n_classes_per_task,
            train=True,
        )
        print(f"Number of classes: {scenario.nb_classes}.")
        print(f"Number of tasks: {scenario.nb_tasks}.")
        if train:
            self.train_datasets.clear()
            self.val_datasets.clear()
        else:
            self.test_datasets.clear()

        for task_id, task_dataset in enumerate(scenario):
            if train:
                train_dataset, val_dataset = split_train_val(task_dataset, val_split=self.val_fraction)
                self.train_datasets.append(train_dataset)
                self.val_datasets.append(val_dataset)
            else:
                test_dataset = task_dataset
                self.test_datasets.append(test_dataset)

    def train_dataloader(self,
                         batch_size: int,
                         num_workers: int = 16,
                         shuffle: bool = True,
                         **kwargs):
        return DataLoader(
            self.train_datasets[self.current_task_id],
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            **kwargs,
        )

    def val_dataloader(self,
                       batch_size: int,
                       num_workers: int = 16,
                       shuffle: bool = True,
                       **kwargs):
        return DataLoader(
            self.val_datasets[self.current_task_id],
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            **kwargs,
        )

    def test_dataloader(self,
                        batch_size: int,
                        num_workers: int = 16,
                        shuffle: bool = True,
                        **kwargs):
        return DataLoader(
            self.test_datasets[self.current_task_id],
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            **kwargs,
        )

if __name__ == "__main__":
    from methods import BaselineMethod
    ClassIncrementalSetting.main()
