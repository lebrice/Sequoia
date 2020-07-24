import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import (Any, Callable, ClassVar, Dict, List, Optional, Tuple, Type,
                    TypeVar, Union)

from torch import Tensor
from torch.utils.data import DataLoader, Dataset

import pl_bolts
from continuum import ClassIncremental, split_train_val
from continuum.datasets import *
from continuum.datasets import _ContinuumDataset
from continuum.scenarios.base import _BaseCLLoader
from datasets.data_utils import FixChannels
from pl_bolts.datamodules import LightningDataModule, MNISTDataModule
from simple_parsing import (Serializable, choice, field, list_field,
                            mutable_field)

from setups.base import Compose, PassiveSetting, Transforms
from setups.environment import (ActionType, ActiveEnvironment, EnvironmentBase,
                          ObservationType, PassiveEnvironment, RewardType)
from utils.logging_utils import get_logger

logger = get_logger(__file__)

num_classes_in_dataset: Dict[str, int] = {
    "mnist": 10,
    "fashion_mnist": 10,
    "kmnist": 10,
    "emnist": 10,
    "qmnist": 10,
    "mnist_fellowship": 30,
    "cifar10": 10,
    "cifar100": 100,
    "cifar_fellowship": 110,
    "imagenet100": 100,
    "imagenet1000": 1000,
    "permuted_mnist": 10,
    "rotated_mnist": 10,
    "core50": 50,
    "core50-v2-79": 50,
    "core50-v2-196": 50,
    "core50-v2-391": 50,
}

dims_for_dataset: Dict[str, Tuple[int, int, int]] = {
    "mnist": (28, 28, 1),
    "fashion_mnist": (28, 28, 1),
    "kmnist": (28, 28, 1),
    "emnist": (28, 28, 1),
    "qmnist": (28, 28, 1),
    "mnist_fellowship": (28, 28, 1),
    "cifar10": (32, 32, 3),
    "cifar100": (32, 32, 3),
    "cifar_fellowship": (32, 32, 3),
    "imagenet100": (224, 224, 3),
    "imagenet1000": (224, 224, 3),
    "permuted_mnist": (28, 28, 1),
    "rotated_mnist": (28, 28, 1),
    "core50": (224, 224, 3),
    "core50-v2-79": (224, 224, 3),
    "core50-v2-196": (224, 224, 3),
    "core50-v2-391": (224, 224, 3),
}
from common.dims import Dims


@dataclass
class ContinualSetting(PassiveSetting[ObservationType, RewardType]):
    """Settings where the data is online non-stationary.

    At the moment, this is basically just a base class for the Class-Incremental
    Setting (ClassIncrementalSetting) where the data is split up into 'tasks'.

    However in the future, as we add more CL setups, they should extend this
    class, and we might need to move some of the stuff here into
    ClassIncremental if needed.

    For example, we might want to create something like a 'stream' learning of
    some sort, where the transitions are smooth and there are no task labels.

    This implements the LightningDataModule API from pytorch-lightning-bolts.
    The hope is that this greatly simplifies the whole data generation process.
    - `train_dataloader`, `val_dataloader` and `test_dataloader` give
        dataloaders of the current task.
    - `train_dataloaders`, `val_dataloaders` and `test_dataloaders` give the 
        dataloaders of all the tasks. 

    The current task can be set at the `current_task_id` attribute.

    TODO: Add the missing members from LightningDataModule
    TODO: Maybe add a way to 'wrap' another LightningDataModule?
    TODO: Change the base class from PassiveSetting to `ActiveSetting` for
    continual active learning / continual RL.
    """

    # Class variable holding all the available datasets.
    available_datasets: ClassVar[Dict[str, Type[_ContinuumDataset]]] = {
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
    # A continual dataset to use. (Should be taken from the continuum package).
    dataset: str = choice(available_datasets.keys(), default="mnist")
    
    # Wether the current task id can be read from outside this class.
    # NOTE: Loosely enforced, could be bypassed if people want to 'cheat'.
    # TODO: Adding a mechanism for making task label only available at train time?
    task_label_is_readable: bool = True
    # Wether the current task id can be set from outside this class.
    task_label_is_writable: bool = True

    # Transformations to use. See the Transforms enum for the available values.
    transforms: List[Transforms] = list_field(Transforms.to_tensor, Transforms.fix_channels, to_dict=False)

    def __post_init__(self):
        """Creates a new CL environment / setup.

        Args:
            options (Options): Dataclass used for configuration.
        """
        super().__post_init__()
        self._current_task_id: int = 0

        self.train_dataset: _ContinuumDataset = None
        self.test_dataset: _ContinuumDataset = None
        self.train_datasets: List[_ContinuumDataset] = []
        self.val_datasets:   List[_ContinuumDataset] = []
        self.test_datasets:  List[_ContinuumDataset] = []
        
        self._dims: Tuple[int, int, int] = dims_for_dataset[self.dataset]
        self._num_classes: int =  num_classes_in_dataset[self.dataset]

    @property
    def dataset_class(self) -> Type[_ContinuumDataset]:
        return type(self).available_datasets[self.dataset]

    def make_dataset(self, data_dir: Path, download: bool=True, train: bool=True, transform: Callable=None, **kwargs) -> _ContinuumDataset:
        return self.dataset_class(
            data_path=data_dir,
            download=download,
            **kwargs
        )

    @property
    def dims(self) -> Dims:
        """Gets the dimensions of the input, taking into account the transforms.
        
        # TODO: Could transforms just specify their impact on the shape directly instead, à-la Tensorflow?
        """
        dims = Dims(*self._dims)
        assert dims.c < dims.h and dims.c < dims.w and dims.h == dims.w, dims

        if Transforms.fix_channels in self.transforms:
            dims = dims._replace(c=3)
            return dims
        return self._dims

    @dims.setter
    def dims(self, value: Any):
        self._dims = value

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @num_classes.setter
    def num_classes(self, value: int) -> None:
        self._num_classes = value

    @abstractmethod
    def make_train_cl_loader(self, dataset: _ContinuumDataset) -> _BaseCLLoader:
        """ Creates a train CL Loader using the continuum package. """
        
    @abstractmethod
    def make_test_cl_loader(self, dataset: _ContinuumDataset) -> _BaseCLLoader:
        """ Creates a test CL Loader using the continuum package.  """

    def prepare_data(self, data_dir: Path, **kwargs):
        """ Prepares data, downloads the dataset, creates the datasets for each
        task.

        TODO: Not supposed to assign stuff to `self` because of DP training.. need to check. 
        """
        self.cl_dataset = self.make_dataset(data_dir, download=True)
        self.train_cl_loader: _BaseCLLoader = self.make_train_cl_loader(self.cl_dataset)
        self.test_cl_loader: _BaseCLLoader = self.make_test_cl_loader(self.cl_dataset)

        logger.info(f"Number of train tasks: {self.train_cl_loader.nb_tasks}.")
        logger.info(f"Number of test tasks: {self.train_cl_loader.nb_tasks}.")

        self.train_datasets.clear()
        self.val_datasets.clear()
        self.test_datasets.clear()
        
        for task_id, train_dataset in enumerate(self.train_cl_loader):
            train_dataset, val_dataset = split_train_val(train_dataset, val_split=self.val_fraction)
            self.train_datasets.append(train_dataset)
            self.val_datasets.append(val_dataset)

        for task_id, test_dataset in enumerate(self.test_cl_loader):
            self.test_datasets.append(test_dataset)

        return super().prepare_data(**kwargs)

    def train_dataloader(self, **kwargs) -> PassiveEnvironment:
        """Returns a DataLoader for the train dataset of the current task.
        
        NOTE: The dataloader is passive for now (just a regular DataLoader).
        """
        dataset = self.train_datasets[self._current_task_id]
        env: DataLoader = PassiveEnvironment(dataset, **kwargs)
        return env

    def val_dataloader(self, **kwargs) -> PassiveEnvironment:
        """Returns a DataLoader for the validation dataset of the current task.
        
        NOTE: The dataloader is passive for now (just a regular DataLoader).
        """
        dataset = self.val_datasets[self._current_task_id]
        env: DataLoader = PassiveEnvironment(dataset, **kwargs)
        return env

    def test_dataloader(self, **kwargs) -> PassiveEnvironment:
        """Returns a DataLoader for the test dataset of the current task.
        
        NOTE: The dataloader is passive for now (just a regular DataLoader).
        """
        dataset = self.test_datasets[self._current_task_id]
        env: DataLoader = PassiveEnvironment(dataset, **kwargs)
        return env

    def train_dataloaders(self, **kwargs) -> List[PassiveEnvironment]:
        """Returns the DataLoaders for all the train datasets. """
        loaders: List[DataLoader] = []
        for i, dataset in enumerate(self.train_datasets):
            env: DataLoader = PassiveEnvironment(dataset, **kwargs)
            loaders.append(env)
        return loaders

    def test_dataloaders(self, **kwargs) -> List[PassiveEnvironment]:
        """Returns the DataLoaders for all the test datasets. """
        loaders: List[DataLoader] = []
        for i, dataset in enumerate(self.test_datasets):
            env: DataLoader = PassiveEnvironment(dataset, **kwargs)
            loaders.append(env)
        return loaders

    def val_dataloaders(self, **kwargs) -> List[PassiveEnvironment]:
        """Returns the DataLoaders for all the validation datasets. """
        loaders: List[DataLoader] = []
        for i, dataset in enumerate(self.val_datasets):
            env: DataLoader = PassiveEnvironment(dataset, **kwargs)
            loaders.append(env)
        return loaders

    @property
    def current_task_id(self) -> Optional[int]:
        """ Get the current task or None when it is not available. """
        if self.task_label_is_readable:
            return self._current_task_id
        else:
            return None
    
    @current_task_id.setter
    def current_task_id(self, value: int) -> None:
        """ Set the current task when it is writable else raises a warning. """
        if self.task_label_is_writable:
            self._current_task_id = value
        else:
            warnings.warn(UserWarning(
                f"Trying to set task id but it is not writable! Doing nothing."
            ))
