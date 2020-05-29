import torch
import numpy as np
from itertools import repeat, cycle
from torch.utils.data.sampler import SubsetRandomSampler
import random
from abc import ABC, abstractmethod
from dataclasses import InitVar, dataclass
from pathlib import Path
from typing import *

import torch
import torchvision
from simple_parsing import choice, field
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets as v_datasets
from torchvision import transforms
from torchvision.utils import save_image

from config import Config

def get_semi_sampler(labels, p:float=None):
    #p - percentage of labeled data to be kept
    #print(type(labels))
    indices = np.arange(len(labels))
    classes = np.unique(labels)
    # Ensure uniform distribution of labels
    np.random.shuffle(indices)

    indices_train = [list(filter(lambda idx: labels[idx] == i, indices))[:] for i in classes]
    indices_train = np.hstack([indices_train[i][:int(p*len(indices_train[i]))] for i in range(len(indices_train))])
    indices_unlabelled = np.hstack(
        [list(filter(lambda idx: labels[idx] == i, indices))[:] for i in classes])
    # print (indices_train.shape)
    # print (indices_valid.shape)
    # print (indices_unlabelled.shape)
    indices_train = torch.from_numpy(indices_train)
    indices_unlabelled = torch.from_numpy(indices_unlabelled)
    sampler_train = SubsetRandomSampler(indices_train)
    sampler_unlabelled = SubsetRandomSampler(indices_unlabelled)
    return sampler_train, sampler_unlabelled

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
    dataset_class: Type[Dataset] = field(default=v_datasets.MNIST, repr=False)

    x_shape: Tuple[int, int, int] = (1, 28, 28)
    y_shape: Tuple[int] = (10,)

    train: Optional[Dataset] = None
    valid: Optional[Dataset] = None

    # TODO: This isn't really actually ever used. The VisionDatasets would call
    # them in __getitem__, and we don't really use that..
    transforms: object = transforms.ToTensor()

    _loaded: bool = False

    def load(self, data_dir: Path = None) -> Tuple[Dataset, Dataset]:
        """ Downloads the corresponding datasets.

        TODO: Maybe figure out a way to get the resizing to happen here instead of
        in Classifier.process_inputs whenever we're using a pretrained encoder model?
        Would there be a benefit in doing so?
        """
        if self._loaded:
            assert self.train, self.valid
            return self.train, self.valid

        # Use the data_dir argument if given, otherwise use "./data"
        data_dir = data_dir or Path("data")
        self.train = self.dataset_class(data_dir, train=True, download=True, transform=self.transforms)
        self.valid = self.dataset_class(data_dir, train=False, download=True, transform=self.transforms)
        self._loaded = True
        return self.train, self.valid

    def get_dataloaders(self, config: Config, batch_size: int = 64) -> Tuple[
        Optional[DataLoader], Optional[DataLoader]]:
        """Create the train and test dataloaders using the passed arguments.

        You might want to override/extend this method subclasses.

        (NOTE: currently unused, will probably be removed at some point.)

        Returns
        -------
        Tuple[DataLoader, DataLoader]
            The training and validation dataloaders.
        """
        self.load(config.log_dir)

        train_loader = None
        if self.train:
            train_loader = DataLoader(
                self.train,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=config.use_cuda,
            )

        valid_loader = None
        if self.valid:
            valid_loader = DataLoader(
                self.valid,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=config.use_cuda,
            )
        return train_loader, valid_loader
