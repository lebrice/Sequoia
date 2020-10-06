""" Defines a `Setting` subclass for "Class-Incremental" Continual Learning.

Example command to run a method on this setting (in debug mode):
```
python main.py --setting class_incremental --method baseline --debug  \
    --batch_size 128 --max_epochs 1
```

TODO: I'm not sure this fits the "Class-Incremental" definition from
[iCaRL](https://arxiv.org/abs/1611.07725) at the moment:

    "Formally, we demand the following three properties of an algorithm to qualify
    as class-incremental:
    i)  it should be trainable from a stream of data in which examples of
        different classes occur at different times
    ii) it should at any time provide a competitive multi-class classifier for
        the classes observed so far,
    iii) its computational requirements and memory footprint should remain
        bounded, or at least grow very slowly, with respect to the number of classes
        seen so far."
"""
import dataclasses
import warnings
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import (Callable, ClassVar, Dict, List, Optional, Sequence, Tuple,
                    Type, Union)

import matplotlib.pyplot as plt
import torch
import numpy as np
from continuum import ClassIncremental
from continuum.datasets import *
from continuum.datasets import _ContinuumDataset
from continuum.scenarios.base import _BaseCLLoader
from continuum.tasks import split_train_val
from pytorch_lightning import LightningModule, Trainer
from simple_parsing import choice, list_field
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import ClassificationMetrics, Metrics, get_metrics
from common.config import Config
from common.loss import Loss
from common.transforms import Transforms, SplitBatch, Compose
from settings.method_abc import MethodABC
from settings.base.results import Results
from settings.base.environment import ObservationType, RewardType
from utils import dict_union, get_logger, constant, mean, take

from .batch_transforms import RelabelTransform, ReorderTensors, DropTaskLabels
from .class_incremental_results import ClassIncrementalResults
from ..passive_setting import PassiveSetting
from ..passive_environment import PassiveEnvironment, Actions, ActionType, Observations, Rewards

from settings.assumptions.incremental import IncrementalSetting

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


@dataclass
class ClassIncrementalSetting(PassiveSetting, IncrementalSetting):
    """Supervised Setting where the data is a sequence of 'tasks'.

    This class is basically is the supervised version of an Incremental Setting
    
    
    The current task can be set at the `current_task_id` attribute.
    """
    
    Results: ClassVar[Type[Results]] = ClassIncrementalResults

    @dataclass(frozen=True)
    class Observations(IncrementalSetting.Observations,
                       PassiveSetting.Observations):
        """Incremental Observations, in a supervised context.""" 
        pass

    # Class variable holding a dict of the names and types of all available
    # datasets.
    available_datasets: ClassVar[Dict[str, Type[_ContinuumDataset]]] = {
        c.__name__.lower(): c
        for c in [
            CIFARFellowship, Fellowship, MNISTFellowship, ImageNet100,
            ImageNet1000, MultiNLI, CIFAR10, CIFAR100, EMNIST, KMNIST, MNIST,
            QMNIST, FashionMNIST,
        ]
    }
    # A continual dataset to use. (Should be taken from the continuum package).
    dataset: str = choice(available_datasets.keys(), default="mnist")
    
    # Transformations to use. See the Transforms enum for the available values.
    transforms: List[Transforms] = list_field(
        Transforms.to_tensor,
        # BUG: The input_shape given to the Model doesn't have the right number
        # of channels, even if we 'fixed' them here. However the images are fine
        # after.
        Transforms.three_channels,
        Transforms.channels_first_if_needed,
    )

    # Either number of classes per task, or a list specifying for
    # every task the amount of new classes.
    increment: Union[int, List[int]] = list_field(2, type=int, nargs="*", alias="n_classes_per_task")
    # The scenario number of tasks.
    # If zero, defaults to the number of classes divied by the increment.
    nb_tasks: int = 0
    # A different task size applied only for the first task.
    # Desactivated if `increment` is a list.
    initial_increment: int = 0
    # An optional custom class order, used for NC.
    class_order: Optional[List[int]] = None
    # Either number of classes per task, or a list specifying for
    # every task the amount of new classes (defaults to the value of
    # `increment`).
    test_increment: Optional[Union[List[int], int]] = None
    # A different task size applied only for the first test task.
    # Desactivated if `test_increment` is a list. Defaults to the
    # value of `initial_increment`.
    test_initial_increment: Optional[int] = None
    # An optional custom class order for testing, used for NC.
    # Defaults to the value of `class_order`.
    test_class_order: Optional[List[int]] = None

    def __post_init__(self):
        """Initializes the fields of the Setting (and LightningDataModule),
        including the transforms, shapes, etc.
        """
        if not hasattr(self, "num_classes"):
            # In some concrete LightningDataModule's like MnistDataModule,
            # num_classes is a read-only property. Therefore we check if it
            # is already defined. This is just in case something tries to
            # inherit from both IIDSetting and MnistDataModule, for instance.
            self.num_classes: int = num_classes_in_dataset[self.dataset]
        if hasattr(self, "dims"):
            # NOTE This sould only happen if we subclass both a concrete
            # LightningDataModule like MnistDataModule and a Setting (e.g.
            # IIDSetting) like above.
            image_shape = self.dims
        else:
            image_shape: Tuple[int, int, int] = dims_for_dataset[self.dataset]

        if isinstance(self.increment, list) and len(self.increment) == 1:
            # This can happen when parsing a list from the command-line.
            self.increment = self.increment[0]

        # Set the number of tasks depending on the increment, and vice-versa.
        # (as only one of the two should be used).
        if self.nb_tasks == 0:
            self.nb_tasks = self.num_classes // self.increment
        else:
            self.increment = self.num_classes // self.nb_tasks

        if not self.class_order:
            self.class_order = list(range(self.num_classes))

        # Test values default to the same as train.
        self.test_increment = self.test_increment or self.increment
        self.test_initial_increment = self.test_initial_increment or self.test_increment
        self.test_class_order = self.test_class_order or self.class_order

        # TODO: For now we assume a fixed, equal number of classes per task, for
        # sake of simplicity. We could take out this assumption, but it might
        # make things a bit more complicated.
        assert isinstance(self.increment, int) and isinstance(self.test_increment, int)
        self.n_classes_per_task: int = self.increment
        
        from gym.spaces import Box, Discrete, Tuple as SpaceTuple, Dict as SpaceDict
        action_shape = (self.n_classes_per_task,)
        self.observation_space = SpaceDict({
            "x": Box(low=0, high=1, shape=image_shape, dtype=np.float32),
            "task_labels": Discrete(self.nb_tasks)
        })
        # TODO: Change the actions from logits to predicted labels.
        # self.action_space = Discrete(self.n_classes_per_task)
        self.action_space = Box(low=-np.inf, high=np.inf, shape=(self.n_classes_per_task,))
        self.reward_space = Discrete(self.num_classes)
        print(self.observation_space)
        print(self.action_space)
        # TODO: Need to change this so the 'actions' are actually the predicted
        # labels, not the logits.
        super().__post_init__(
            observation_space=self.observation_space,
            action_space=self.action_space,
            reward_space=self.reward_space, # the labels have shape (1,) always.
        )

        self.train_datasets: List[_ContinuumDataset] = []
        self.val_datasets: List[_ContinuumDataset] = []
        self.test_datasets: List[_ContinuumDataset] = []
        
        # This will be set by the Experiment, or passed to the `apply` method.
        # TODO: This could be a bit cleaner.
        self.config: Config
        # Default path to which the datasets will be downloaded.
        self.data_dir: Optional[Path] = None

    def apply(self, method: "Method", config: Config) -> ClassIncrementalResults:
        """Apply the given method on this setting to producing some results."""
        # NOTE: (@lebrice) The test loop is written by hand here because I don't
        # want to have to give the labels to the method at test-time. See the
        # docstring of `test_loop` for more info.
        from methods import Method
        method: Method

        self.config = config
        method.config = config
        # TODO: At the moment, we're nice enough to do this, but this would
        # maybe allow the method to "cheat"!
        method.configure(setting=self)
        self.configure(method)                
        # Run the Training loop (which is defined in IncrementalSetting).
        self.train_loop(method)
        
        results: ClassIncrementalResults = self.test_loop(method)
        logger.info(f"Resulting objective of Test Loop: {results.objective}")
        print(results.summary())
        log_dict = results.to_log_dict()
        results.save_to_dir(self.config.log_dir)
        return results

    def prepare_data(self, data_dir: Path = None, **kwargs):
        data_dir = data_dir or self.data_dir or self.config.data_dir
        self.make_dataset(data_dir, download=True)
        self.data_dir = data_dir
        super().prepare_data(**kwargs)

    def setup(self, stage: Optional[str] = None, *args, **kwargs):
        """ Creates the datasets for each task.
        
        TODO: Figure out a way of setting data_dir elsewhere maybe?
        """
        logger.info(f"data_dir: {self.data_dir}, setup args: {args} kwargs: {kwargs}")
        
        self.train_cl_dataset = self.make_dataset(self.data_dir, download=False, train=True)
        self.test_cl_dataset = self.make_dataset(self.data_dir, download=False, train=False)
        self.train_cl_loader: _BaseCLLoader = self.make_train_cl_loader(self.train_cl_dataset)
        self.test_cl_loader: _BaseCLLoader = self.make_test_cl_loader(self.test_cl_dataset)

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

        super().setup(stage, *args, **kwargs)

    def train_dataloader(self, **kwargs) -> PassiveEnvironment:
        """Returns a DataLoader for the train dataset of the current task. """
        if not self.has_prepared_data:
            self.prepare_data()
        if not self.has_setup_fit:
            self.setup("fit")
        dataset = self.train_datasets[self.current_task_id]
        # TODO: Add some kind of Wrapper around the dataset to make it
        # semi-supervised.
        kwargs = dict_union(self.dataloader_kwargs, kwargs)
        batch_transforms = self.train_batch_transforms()
        self.train_env = PassiveEnvironment(
            dataset,
            batch_transforms=batch_transforms,
            observations_type=self.Observations,
            actions_type=self.Actions,
            rewards_type=self.Rewards,
            **kwargs,
        )
        return self.train_env

    def val_dataloader(self, **kwargs) -> PassiveEnvironment:
        """Returns a DataLoader for the validation dataset of the current task.
        """
        if not self.has_prepared_data:
            self.prepare_data()
        if not self.has_setup_fit:
            self.setup("fit")
        dataset = self.val_datasets[self.current_task_id]
        kwargs = dict_union(self.dataloader_kwargs, kwargs)
        
        batch_transforms: List[Callable] = self.val_batch_transforms()
        self.val_env = PassiveEnvironment(
            dataset,
            batch_transforms=batch_transforms,
            observations_type=self.Observations,
            actions_type=self.Actions,
            rewards_type=self.Rewards,
            **kwargs,
        )
        return self.val_env

    def test_dataloader(self, **kwargs) -> PassiveEnvironment["ClassIncrementalSetting.Observations", Actions, Rewards]:
        """Returns a DataLoader for the test dataset of the current task.
        """
        if not self.has_prepared_data:
            self.prepare_data()
        if not self.has_setup_test:
            self.setup("test")
        dataset = self.test_datasets[self.current_task_id]
        kwargs = dict_union(self.dataloader_kwargs, kwargs)
        
        batch_transforms: List[Callable] = self.test_batch_transforms()
        self.test_env = PassiveEnvironment(
            dataset,
            batch_transforms=batch_transforms,
            observations_type=self.Observations,
            actions_type=self.Actions,
            rewards_type=self.Rewards,
            # TODO: Enabling this at test time, just for fun. This means that we
            # could perhaps just keep track of the metrics in the testing environment!
            pretend_to_be_active=True,
            **kwargs,
        )
        return self.test_env

    def configure(self, method: MethodABC):
        """ Setup the data_dir and the dataloader kwargs using properties of the
        Method or of self.Config.

        Parameters
        ----------
        method : MethodABC
            The Method that is being applied on this setting.
        config : Config
            [description]
        """
        assert self.config is not None
        config = self.config
        # Get the arguments that will be used to create the dataloaders.
        
        # TODO: Should the data_dir be in the Setting, or the Config?
        self.data_dir = config.data_dir
        
        # Create the dataloader kwargs, if needed.
        if not self.dataloader_kwargs:
            batch_size = 32
            if hasattr(method, "batch_size"):
                batch_size = method.batch_size
            elif hasattr(method, "model") and hasattr(method.model, "batch_size"):
                batch_size = method.model.batch_size
            elif hasattr(config, "batch_size"):
                batch_size = config.batch_size

            dataloader_kwargs = dict(
                batch_size=batch_size,
                num_workers=config.num_workers,
                shuffle=False,
            )
        # Save the dataloader kwargs in `self` so that calling `train_dataloader()`
        # from outside with no arguments (i.e. when fitting the model with self
        # as the datamodule) will use the same args as passing the dataloaders
        # manually.
        self.dataloader_kwargs = dataloader_kwargs
        logger.debug(f"Dataloader kwargs: {dataloader_kwargs}")

        # Debugging: Run a quick check to see that what is returned by the
        # dataloaders is of the right type and shape etc.
        self._check_dataloaders_give_correct_types()

    def make_train_cl_loader(self, train_dataset: _ContinuumDataset) -> _BaseCLLoader:
        """ Creates a train ClassIncremental object from continuum. """
        return ClassIncremental(
            train_dataset,
            nb_tasks=self.nb_tasks,
            increment=self.increment,
            initial_increment=self.initial_increment,
            class_order=self.class_order,
            transformations=self.train_transforms,
        )

    def make_test_cl_loader(self, test_dataset: _ContinuumDataset) -> _BaseCLLoader:
        """ Creates a test ClassIncremental object from continuum. """
        return ClassIncremental(
            test_dataset,
            nb_tasks=self.nb_tasks,
            increment=self.test_increment,
            initial_increment=self.test_initial_increment,
            class_order=self.test_class_order,
            transformations=self.test_transforms,
        )

    @property
    def dataset_class(self) -> Type[_ContinuumDataset]:
        return self.available_datasets[self.dataset]

    def make_dataset(self,
                     data_dir: Path,
                     download: bool = True,
                     train: bool = True,
                     **kwargs) -> _ContinuumDataset:
        # TODO: #7 Use this method here to fix the errors that happen when
        # trying to create every single dataset from continuum. 
        return self.dataset_class(
            data_path=data_dir,
            download=download,
            train=train,
            **kwargs
        )

    def train_batch_transforms(self) -> List[Callable]:
        transforms = [
            RelabelTransform(task_classes=self.current_task_classes(train=True)),
            ReorderTensors(), # (x, y, t) -> (x, t, y)
            SplitBatch(self.Observations, self.Rewards),
        ]
        if not self.task_labels_at_train_time:
            transforms.append(DropTaskLabels())
        return transforms
    
    def val_batch_transforms(self) -> List[Callable]:
        return self.train_batch_transforms()
    
    def test_batch_transforms(self) -> List[Callable]:
        transforms = [
            RelabelTransform(task_classes=self.current_task_classes(train=False)),
            ReorderTensors(), # (x, y, t) -> (x, (t), y)
            SplitBatch(self.Observations, self.Rewards),
        ]
        if not self.task_labels_at_test_time:
            transforms.append(DropTaskLabels())
        return transforms


    def _check_dataloaders_give_correct_types(self):
        """ Do a quick check to make sure that the dataloaders give back the
        right observations / reward types.
        """
        for loader_method in [self.train_dataloader, self.val_dataloader, self.test_dataloader]:
            env = loader_method()
            for observations, *rewards in take(env, 5):
                try:
                    assert isinstance(observations, self.Observations), type(observations)
                    rewards: Optional[Rewards] = rewards[0] if rewards else None
                    if rewards is not None:
                        assert isinstance(rewards, self.Rewards), type(rewards)
                    # TODO: If we add gym spaces to all environments, then check
                    # that the observations are in the observation space, sample
                    # a random action from the action space, check that it is
                    # contained within that space, and then get a reward by
                    # sending it to the dataloader. Check that the reward
                    # received is in the reward space.
                    
                    actions = self.Actions(torch.rand([observations.batch_size, self.n_classes_per_task]))
                    rewards = env.send(actions)
                    assert isinstance(rewards, self.Rewards), type(rewards)
                except Exception as e:
                    logger.error(f"There's a problem with the method {loader_method} (env {env})")
                    raise e
    
    # These methods below are used by the ClassIncrementalModel, mostly when
    # using a multihead model, to figure out how to relabel the batches, or how
    # many classes there are in the current task (since we support a different
    # number of classes per task).
    # TODO: Remove this? Since I'm simplifying to a fixed number of classes per
    # task for now... 

    def num_classes_in_task(self, task_id: int, train: bool) -> Union[int, List[int]]:
        """ Returns the number of classes in the given task. """
        increment = self.increment if train else self.test_increment
        if isinstance(increment, list):
            return increment[task_id]
        return increment

    def num_classes_in_current_task(self, train: bool) -> int:
        """ Returns the number of classes in the current task. """
        return self.num_classes_in_task(self._current_task_id, train=train)

    def task_classes(self, task_id: int, train: bool) -> List[int]:
        """ Gives back the 'true' labels present in the given task. """
        start_index = sum(
            self.num_classes_in_task(i, train) for i in range(task_id)
        )
        end_index = start_index + self.num_classes_in_task(task_id, train)
        if train:
            return self.class_order[start_index:end_index]
        else:
            return self.test_class_order[start_index:end_index]

    def current_task_classes(self, train: bool) -> List[int]:
        """ Gives back the labels present in the current task. """
        return self.task_classes(self._current_task_id, train)
    
    def relabel(self, y: Tensor, train: bool):
        # Re-label the given batch so the losses/metrics work correctly.
        # Example: if the current task classes is [2, 3] then relabel that
        # those examples as [0, 1].
        # TODO: Double-check that that this is what is usually done in CL.
        new_y = torch.empty_like(y)
        for i, label in enumerate(self.current_task_classes(train)):
            new_y[y == label] = i
        return new_y
