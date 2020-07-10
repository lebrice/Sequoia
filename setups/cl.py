import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import (Callable, ClassVar, Dict, List, Optional, Tuple, Type,
                    TypeVar, Union)

from torch import Tensor
from torch.utils.data import DataLoader, Dataset

import pl_bolts
from continuum import ClassIncremental, split_train_val
from continuum.datasets import *
from continuum.datasets import _ContinuumDataset
from continuum.scenarios.base import _BaseCLLoader
from pl_bolts.datamodules import LightningDataModule, MNISTDataModule
from simple_parsing import Serializable, choice, list_field, mutable_field

from .base import PassiveSetting
from .environment import (ActionType, ActiveEnvironment, EnvironmentBase,
                          ObservationType, PassiveEnvironment, RewardType)

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
    "mnist": [28, 28, 1],
    "fashion_mnist": [28, 28, 1],
    "kmnist": [28, 28, 1],
    "emnist": [28, 28, 1],
    "qmnist": [28, 28, 1],
    "mnist_fellowship": [28, 28, 1],
    "cifar10": [32, 32, 3],
    "cifar100": [32, 32, 3],
    "cifar_fellowship": [32, 32, 3],
    "imagenet100": [224, 224, 3],
    "imagenet1000": [224, 224, 3],
    "permuted_mnist": [28, 28, 1],
    "rotated_mnist": [28, 28, 1],
    "core50": [224, 224, 3],
    "core50-v2-79": [224, 224, 3],
    "core50-v2-196": [224, 224, 3],
    "core50-v2-391": [224, 224, 3],
}

@dataclass
class CLSetting(PassiveSetting[ObservationType, RewardType]):
    """LightningDataModule for CL experiments.

    This greatly simplifies the whole data generation process.
    the train_dataloader, val_dataloader and test_dataloader methods are used
    to get the dataloaders of the current task.

    The current task can be set at the `current_task_id` attribute.

    TODO: Add the missing members from LightningDataModule
    TODO: Maybe add a way to 'wrap' another LightningDataModule?
    TODO: Change the base class from PassiveSetting to `ActiveSetting` for
    continual active learning / continual RL.
    """

    @dataclass
    class Config(PassiveSetting.Config):
        """ Config for the environment. """
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
        # fraction of training data to devote to validation. Defaults to 0.2.
        val_fraction: float = 0.2
        # Wether the current task id can be read from outside this class.
        # NOTE: Loosely enforced, could be bypassed if people want to 'cheat'.
        # TODO: Adding a mechanism for making task label only available at test time?
        task_label_is_readable: bool = True
        # Wether the current task id can be set from outside this class.
        task_label_is_writable: bool = True

        @property
        def dataset_class(self) -> Type[_ContinuumDataset]:
            return type(self).available_datasets[self.dataset]

    # Configuration options for the environment / setup / datasets.
    config: Config = None

    def __init__(self, config: Config = None):
        """Creates a new CL environment / setup.

        Args:
            config (Config): Dataclass used for configuration.
        """
        super().__init__(config=config or self.Config())
        self.config: self.Config
        self.val_fraction: float = config.val_fraction
        
        self.__task_label_is_readable: bool = config.task_label_is_readable
        self.__task_label_is_writable: bool = config.task_label_is_writable
        self.__current_task_id: int = 0

        self.dataset: _ContinuumDataset = None
        self.train_datasets: List[_ContinuumDataset] = []
        self.val_datasets:   List[_ContinuumDataset] = []
        self.test_datasets:  List[_ContinuumDataset] = []

    @property
    def dims(self) -> Tuple[int, int, int]:
        return dims_for_dataset[self.config.dataset]

    @property
    def num_classes(self) -> int:
        return num_classes_in_dataset[self.config.dataset]

    @abstractmethod
    def make_train_cl_loader(self) -> _BaseCLLoader:
        """ Creates a train CL Loader using the continuum package. """
        
    @abstractmethod
    def make_test_cl_loader(self) -> _BaseCLLoader:
        """ Creates a test CL Loader using the continuum package.  """

    def prepare_data(self, *args, data_dir: Path="data", **kwargs):
        """ Prepares data, downloads the dataset, creates the datasets for each
        task.
        """
        self.dataset: _ContinuumDataset = self.config.dataset_class(data_path=data_dir, download=True)
        self.train_cl_loader: _BaseCLLoader = self.make_test_cl_loader()
        self.test_cl_loader: _BaseCLLoader = self.make_test_cl_loader()

        print(f"Number of train classes: {self.train_cl_loader.nb_classes}.")
        print(f"Number of train tasks: {self.train_cl_loader.nb_tasks}.")
        print(f"Number of test classes: {self.train_cl_loader.nb_classes}.")
        print(f"Number of test tasks: {self.train_cl_loader.nb_tasks}.")
        self.train_datasets.clear()
        self.val_datasets.clear()
        self.test_datasets.clear()
        
        for task_id, train_dataset in enumerate(self.train_cl_loader):
            train_dataset, val_dataset = split_train_val(train_dataset, val_split=self.val_fraction)
            self.train_datasets.append(train_dataset)
            self.val_datasets.append(val_dataset)

        for task_id, test_dataset in enumerate(self.test_cl_loader):
            self.test_datasets.append(test_dataset)

        return super().prepare_data(*args, **kwargs)

    def train_dataloader(self, *args, **kwargs) -> PassiveEnvironment:
        dataset = self.train_datasets[self.__current_task_id]
        return PassiveEnvironment(dataset, *args, **kwargs)

    def val_dataloader(self, *args, **kwargs) -> PassiveEnvironment:
        dataset = self.val_datasets[self.__current_task_id]
        return PassiveEnvironment(dataset, *args, **kwargs)

    def test_dataloader(self, *args, **kwargs) -> PassiveEnvironment:
        dataset = self.test_datasets[self.__current_task_id]
        return PassiveEnvironment(dataset, *args, **kwargs)

    @property
    def current_task_id(self) -> Optional[int]:
        """ Get the current task or None when it is not available. """
        if self.__task_label_is_readable:
            return self.__current_task_id
        else:
            return None
    
    @current_task_id.setter
    def current_task_id(self, value: int) -> None:
        """ Set the current task when it is writable else raises a warning. """
        if self.__task_label_is_writable:
            self.__current_task_id = value
        else:
            warnings.warn(UserWarning(
                f"Trying to set task id but it is not writable! Doing nothing."
            ))

class ClassIncrementalSetting(CLSetting[Tensor, Tensor]):
    """ LightningDataModule for CL. 
    """
    @dataclass
    class Config(CLSetting.Config):
        """ Config for Class Incremental Environment.
        
        'docstrings' were taken from the continuum documentation for the
        ClassIncremental class.        
        """
        # The scenario number of tasks.
        # NOTE: For now same number of train and test tasks.
        nb_tasks: int = 0
        # Either number of classes per task, or a list specifying for
        # every task the amount of new classes.
        increment: Union[List[int], int] = list_field(0, type=int, nargs="*")
        # A different task size applied only for the first task.
        # Desactivated if `increment` is a list.
        initial_increment: int = 0
        # An optional custom class order, used for NC.
        class_order = None
        # Either number of classes per task, or a list specifying for
        # every task the amount of new classes (defaults to the value of
        # `increment`).
        test_increment: Union[List[int], int] = None
        # A different task size applied only for the first test task.
        # Desactivated if `test_increment` is a list. Defaults to the
        # value of `initial_increment`.
        test_initial_increment: int = None
        # An optional custom class order for testing, used for NC.
        # Defaults to the value of `class_order`.
        test_class_order = None

        def __post_init__(self, *args, **kwargs):
            super().__post_init__(*args, **kwargs)
            if len(self.increment) == 1:
                self.increment = self.increment[0]
            self.test_increment = self.test_increment or self.increment
            self.test_initial_increment = self.test_initial_increment or self.test_increment
            self.test_class_order = self.test_class_order or self.class_order

    config: Config = mutable_field(Config)

    def __init__(self, config: Config=None):
        """Creates a ClassIncremental CL LightningDataModule.

        Args:
            config
        """
        super().__init__(config=config or self.Config())
        self.nb_tasks = self.config.nb_tasks
    

    @abstractmethod
    def make_train_cl_loader(self) -> _BaseCLLoader:
        """ Creates a train ClassIncremental object from continuum. """
        # TODO: Should we pass the common_transforms and test_transforms to cl loader?
        return ClassIncremental(
            self.dataset,
            nb_tasks=self.nb_tasks,
            increment=self.increment,
            initial_increment=self.initial_increment,
            class_order=self.class_order,
            train=True  # a different loader for test
        )

    @abstractmethod
    def make_test_cl_loader(self) -> _BaseCLLoader:
        """ Creates a test ClassIncremental object from continuum. """
        return ClassIncremental(
            self.dataset,
            nb_tasks=self.config.nb_tasks, 
            increment=self.test_increment,
            initial_increment=self.test_initial_increment,
            class_order=self.test_class_order,
            train=True  # a different loader for test
        )