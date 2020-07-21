from dataclasses import dataclass
from typing import List, Union, Optional

from torch import Tensor

from continuum import ClassIncremental, split_train_val
from continuum.datasets import *
from continuum.datasets import _ContinuumDataset
from continuum.scenarios.base import _BaseCLLoader
from simple_parsing import list_field
from setups.environment import PassiveEnvironment
from .base import CLSetting, num_classes_in_dataset


@dataclass
class ClassIncrementalSetting(CLSetting[Tensor, Tensor]):
    """ LightningDataModule for CL. 
        
    'docstrings' were taken from the continuum documentation for the
    ClassIncremental class.        
    """
    # The scenario number of tasks.
    # NOTE: For now same number of train and test tasks.
    nb_tasks: int = 0
    # Either number of classes per task, or a list specifying for
    # every task the amount of new classes.
    increment: Union[List[int], int] = list_field(2, type=int, nargs="*", alias="num_classes_per_task")
    # A different task size applied only for the first task.
    # Desactivated if `increment` is a list.
    initial_increment: int = 0
    # An optional custom class order, used for NC.
    class_order: Optional[List[int]] = None
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

    def __post_init__(self):
        """Creates a ClassIncremental CL LightningDataModule.

        Args:
            config
        """
        super().__post_init__()
        if isinstance(self.increment, list) and len(self.increment) == 1:
            self.increment = self.increment[0]
        if self.nb_tasks == 0:
            self.nb_tasks = num_classes_in_dataset[self.dataset] // self.increment
        # Test values default to the same as train.
        self.test_increment = self.test_increment or self.increment
        self.test_initial_increment = self.test_initial_increment or self.test_increment
        self.test_class_order = self.test_class_order or self.class_order

    @property
    def num_classes_per_task(self) -> Union[int, List[int]]:
        return self.increment

    def make_train_cl_loader(self, dataset: _ContinuumDataset) -> _BaseCLLoader:
        """ Creates a train ClassIncremental object from continuum. """
        # TODO: Should we pass the common_transforms and test_transforms to cl loader?
        return ClassIncremental(
            dataset,
            nb_tasks=self.nb_tasks,
            increment=self.increment,
            initial_increment=self.initial_increment,
            class_order=self.class_order,
            common_transformations=self.transforms,
            train_transformations=self.train_transforms,
            # TODO: Learn how to use train_transformations and common_transformations of Continuum?
            # train_transformations=self.train_transforms
            train=True  # a different loader for test
        )

    def make_test_cl_loader(self, dataset: _ContinuumDataset) -> _BaseCLLoader:
        """ Creates a test ClassIncremental object from continuum. """
        return ClassIncremental(
            dataset,
            nb_tasks=self.nb_tasks,
            increment=self.test_increment,
            initial_increment=self.test_initial_increment,
            class_order=self.test_class_order,
            common_transformations=self.transforms,
            train=False  # a different loader for test
        )

    def train_dataloader(self, *args, **kwargs) -> PassiveEnvironment:
        """Returns a DataLoader for the train dataset of the current task.
        
        NOTE: The dataloader is passive for now (just a regular DataLoader).
        """
        dataset = self.train_datasets[self._current_task_id]
        env: DataLoader = PassiveEnvironment(dataset, *args, **kwargs)
        return env

    def val_dataloader(self, *args, **kwargs) -> PassiveEnvironment:
        """Returns a DataLoader for the validation dataset of the current task.
        
        NOTE: The dataloader is passive for now (just a regular DataLoader).
        """
        dataset = self.val_datasets[self._current_task_id]
        env: DataLoader = PassiveEnvironment(dataset, *args, **kwargs)
        return env

    def test_dataloader(self, *args, **kwargs) -> List[PassiveEnvironment]:
        """Returns a list of DataLoaders, one for each of the test datasets.
        
        NOTE: The dataloader is passive for now (just a regular DataLoader).
        """
        loaders: List[DataLoader] = []
        for i, dataset in enumerate(self.test_datasets):
            env: DataLoader = PassiveEnvironment(dataset, *args, **kwargs)
            loaders.append(env)
        return loaders