import random
from abc import ABC, abstractmethod
from dataclasses import InitVar, dataclass
from pathlib import Path
from typing import *

import numpy as np
import torch
import torchvision
from simple_parsing import choice, field, mutable_field
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets as v_datasets
from torchvision import transforms
from torchvision.datasets import VisionDataset
from torchvision.utils import save_image

from config import Config


@dataclass
class DatasetConfig:
    """
    Represents all logic related to a Dataset.

    NOTE (Fabrice): No command-line arguments are created here. Instead, I
    decided to have the arguments related to the dataset be in `config`.
    This is because it was getting a bit too complicated to call the training
    script, for instance it was like `"python main.py task-incremental mnist --debug"`
    I think keeping it at one positional argument only and using `"--dataset mnist"` is nicer.  

    """
    name: str = "default"

    # which dataset class to use. (TODO: add more of them.)
    dataset_class: ClassVar[Type[Dataset]] = field(default=v_datasets.MNIST, repr=False)

    x_shape: Tuple[int, int, int] = (1, 28, 28)
    y_shape: Tuple[int] = (10,)
    
    train: Optional[Dataset] = None
    test: Optional[Dataset] = None

    # TODO: This isn't really actually ever used. The VisionDatasets would call
    # them in __getitem__, and we don't really use that..
    transforms: ClassVar[object] = transforms.ToTensor()

    _loaded: bool = False

    def load(self, data_dir: Path=None) -> Tuple[Dataset, Dataset]:
        """ Downloads the corresponding datasets.

        TODO: Maybe figure out a way to get the resizing to happen here instead of
        in Classifier.process_inputs whenever we're using a pretrained encoder model?
        Would there be a benefit in doing so?
        """
        if self._loaded:
            assert self.train, self.test
            return self.train, self.test

        # Use the data_dir argument if given, otherwise use "./data"
        data_dir = data_dir or Path("data")
        self.train = self.dataset_class(data_dir, train=True,  download=True, transform=self.transforms)
        self.test  = self.dataset_class(data_dir, train=False, download=True, transform=self.transforms)
        
        fix_vision_dataset(self.train)
        fix_vision_dataset(self.test)

        self._loaded = True
        return self.train, self.test

def fix_vision_dataset(dataset: VisionDataset) -> None:
    if not isinstance(dataset.data, (np.ndarray, Tensor)):
        dataset.data = torch.as_tensor(dataset.data)
    if not isinstance(dataset.targets, (np.ndarray, Tensor)):
        dataset.targets = torch.as_tensor(dataset.targets)

    if isinstance(dataset, v_datasets.CIFAR100):
        # TODO: Cifar100 seems to want its 'data' to a numpy ndarray. 
        dataset.data = np.asarray(dataset.data)