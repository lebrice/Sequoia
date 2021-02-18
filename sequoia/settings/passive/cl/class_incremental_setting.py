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
import itertools
import warnings
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import (Callable, ClassVar, Dict, List, Optional, Sequence, Tuple,
                    Type, Union)

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from continuum import ClassIncremental
from continuum.datasets import *
from continuum.datasets import _ContinuumDataset
from continuum.scenarios.base import _BaseScenario
from continuum.tasks import split_train_val
from gym import Space, spaces
from pytorch_lightning import LightningModule, Trainer
from simple_parsing import choice, field, list_field
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm

from sequoia.common import ClassificationMetrics, Metrics, get_metrics
from sequoia.common.config import Config
from sequoia.common.gym_wrappers import TransformObservation
from sequoia.common.gym_wrappers.batch_env.tile_images import tile_images
from sequoia.common.loss import Loss
from sequoia.common.spaces import Image, Sparse
from sequoia.common.spaces.named_tuple import NamedTupleSpace
from sequoia.common.transforms import Compose, SplitBatch, Transforms
from sequoia.settings.assumptions.incremental import (IncrementalSetting,
                                                      TestEnvironment)
from sequoia.settings.base import Method, ObservationType, Results, RewardType
from sequoia.utils import constant, dict_union, get_logger, mean, take

from ..passive_environment import (Actions, ActionType, Observations,
                                   PassiveEnvironment, Rewards)
from ..passive_setting import PassiveSetting
from .class_incremental_results import ClassIncrementalResults

logger = get_logger(__file__)

num_classes_in_dataset: Dict[str, int] = {
    "mnist": 10,
    "fashionmnist": 10,
    "kmnist": 10,
    "emnist": 10,
    "qmnist": 10,
    "mnistfellowship": 30,
    "cifar10": 10,
    "cifar100": 100,
    "cifarfellowship": 110,
    "imagenet100": 100,
    "imagenet1000": 1000,
    "permutedmnist": 10,
    "rotatedmnist": 10,
    "core50": 50,
    "core50-v2-79": 50,
    "core50-v2-196": 50,
    "core50-v2-391": 50,
    "synbols": 48,
}


dims_for_dataset: Dict[str, Tuple[int, int, int]] = {
    "mnist": (1, 28, 28),
    "fashionmnist": (1, 28, 28),
    "kmnist": (1, 28, 28),
    "emnist": (1, 28, 28),
    "qmnist": (1, 28, 28),
    "mnistfellowship": (1, 28, 28),
    "cifar10": (3, 32, 32),
    "cifar100": (3, 32, 32),
    "cifarfellowship": (3, 32, 32),
    "imagenet100": (3, 224, 224),
    "imagenet1000": (3, 224, 224),
    # "permutedmnist": (28, 28, 1),
    # "rotatedmnist": (28, 28, 1),
    "core50": (3, 224, 224),
    "core50-v2-79": (3, 224, 224),
    "core50-v2-196": (3, 224, 224),
    "core50-v2-391": (3, 224, 224),
    "synbols": (3, 224, 224),
}

from sequoia.common.gym_wrappers.convert_tensors import add_tensor_support

# NOTE: This dict reflects the observation space of the different datasets
# *BEFORE* any transforms are applied. The resulting property on the Setting is
# based on this 'base' observation space, passed through the transforms.

base_observation_spaces: Dict[str, Space] = {
    dataset_name: add_tensor_support(Image(0, 1, image_shape, np.float32))
    for dataset_name, image_shape in
    {
        "mnist": (1, 28, 28),
        "fashionmnist": (1, 28, 28),
        "kmnist": (28, 28, 1),
        "emnist": (28, 28, 1),
        "qmnist": (28, 28, 1),
        "mnistfellowship": (28, 28, 1),
        "cifar10": (32, 32, 3),
        "cifar100": (32, 32, 3),
        "cifarfellowship": (32, 32, 3),
        "imagenet100": (224, 224, 3),
        "imagenet1000": (224, 224, 3),
        # "permutedmnist": (28, 28, 1),
        # "rotatedmnist": (28, 28, 1),
        "core50": (224, 224, 3),
        "core50-v2-79": (224, 224, 3),
        "core50-v2-196": (224, 224, 3),
        "core50-v2-391": (224, 224, 3),
    "synbols": (224, 224, 3),
    }.items()
}

reward_spaces: Dict[str, Space] = {
    "mnist": spaces.Discrete(10),
    "fashionmnist": spaces.Discrete(10),
    "kmnist": spaces.Discrete(10),
    "emnist": spaces.Discrete(10),
    "qmnist": spaces.Discrete(10),
    "mnistfellowship": spaces.Discrete(30),
    "cifar10": spaces.Discrete(10),
    "cifar100": spaces.Discrete(100),
    "cifarfellowship": spaces.Discrete(110),
    "imagenet100": spaces.Discrete(100),
    "imagenet1000": spaces.Discrete(1000),
    "permutedmnist": spaces.Discrete(10),
    "rotatedmnist": spaces.Discrete(10),
    "core50": spaces.Discrete(50),
    "core50-v2-79": spaces.Discrete(50),
    "core50-v2-196": spaces.Discrete(50),
    "core50-v2-391": spaces.Discrete(50),
    "synbols": spaces.Discrete(48),
}


@dataclass
class ClassIncrementalSetting(PassiveSetting, IncrementalSetting):
    """Supervised Setting where the data is a sequence of 'tasks'.

    This class is basically is the supervised version of an Incremental Setting
    
    
    The current task can be set at the `current_task_id` attribute.
    """
    
    Results: ClassVar[Type[Results]] = ClassIncrementalResults

    # (NOTE: commenting out PassiveSetting.Observations as it is the same class
    # as Setting.Observations, and we want a consistent method resolution order.
    @dataclass(frozen=True)
    class Observations(#PassiveSetting.Observations,
                       IncrementalSetting.Observations):
        """ Incremental Observations, in a supervised context. """
        pass

    # @dataclass(frozen=True)
    # class Actions(PassiveSetting.Actions,
    #               IncrementalSetting.Actions):
    #     """Incremental Actions, in a supervised (passive) context.""" 
    #     pass

    # @dataclass(frozen=True)
    # class Rewards(PassiveSetting.Rewards,
    #               IncrementalSetting.Rewards):
    #     """Incremental Rewards, in a supervised context.""" 
    #     pass

    # Class variable holding a dict of the names and types of all available
    # datasets.
    # TODO: Issue #43: Support other datasets than just classification
    available_datasets: ClassVar[Dict[str, Type[_ContinuumDataset]]] = {
        c.__name__.lower(): c
        for c in [
            CIFARFellowship, MNISTFellowship, ImageNet100,
            ImageNet1000, CIFAR10, CIFAR100, EMNIST, KMNIST, MNIST,
            QMNIST, FashionMNIST, Synbols,
        ]
        # "synbols": Synbols,
        # "synbols_font": partial(Synbols, task="fonts"),
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

    # TODO: Need to put num_workers in only one place.
    batch_size: int = field(default=32, cmd=False)
    num_workers: int = field(default=4, cmd=False)

    # Wether or not to relabel the images to be within the [0, n_classes_per_task]
    # range. Floating (False by default) in Class-Incremental Setting, but set to True
    # in domain_incremental Setting.
    relabel: bool = False
    
    def __post_init__(self):
        """Initializes the fields of the Setting (and LightningDataModule),
        including the transforms, shapes, etc.
        """
        if isinstance(self.increment, list) and len(self.increment) == 1:
            # This can happen when parsing a list from the command-line.
            self.increment = self.increment[0]

        base_observations_space = base_observation_spaces[self.dataset]
        base_reward_space = reward_spaces[self.dataset]
        # action space = reward space by default
        base_action_space = base_reward_space
        
        if isinstance(base_action_space, spaces.Discrete):
            # Classification dataset

            self.num_classes = base_action_space.n
            # Set the number of tasks depending on the increment, and vice-versa.
            # (as only one of the two should be used).
            if self.nb_tasks == 0:
                self.nb_tasks = self.num_classes // self.increment
            else:
                self.increment = self.num_classes // self.nb_tasks
        else:
            raise NotImplementedError(f"TODO: (issue #43)")
        
        
        if not self.class_order:
            self.class_order = list(range(self.num_classes))

        # Test values default to the same as train.
        self.test_increment = self.test_increment or self.increment
        self.test_initial_increment = self.test_initial_increment or self.test_increment
        self.test_class_order = self.test_class_order or self.class_order

        # TODO: For now we assume a fixed, equal number of classes per task, for
        # sake of simplicity. We could take out this assumption, but it might
        # make things a bit more complicated.
        assert isinstance(self.increment, int)
        assert isinstance(self.test_increment, int)

        self.n_classes_per_task: int = self.increment
        action_space = spaces.Discrete(self.n_classes_per_task)
        reward_space = spaces.Discrete(self.n_classes_per_task)

        super().__post_init__(
            # observation_space=observation_space,
            action_space=action_space,
            reward_space=reward_space, # the labels have shape (1,) always.
        )
        self.train_datasets: List[_ContinuumDataset] = []
        self.val_datasets: List[_ContinuumDataset] = []
        self.test_datasets: List[_ContinuumDataset] = []
        
        # This will be set by the Experiment, or passed to the `apply` method.
        # TODO: This could be a bit cleaner.
        self.config: Config
        # Default path to which the datasets will be downloaded.
        self.data_dir: Optional[Path] = None
        
        self.train_env: PassiveEnvironment = None  # type: ignore
        self.val_env: PassiveEnvironment = None  # type: ignore
        self.test_env: PassiveEnvironment = None  # type: ignore
        

    @property
    def observation_space(self) -> NamedTupleSpace:
        """ The un-batched observation space, based on the choice of dataset and
        the transforms at `self.transforms` (which apply to the train/valid/test
        environments).
        
        The returned spaces is a NamedTupleSpace, with the following properties:
        - `x`: observation space (e.g. `Image` space)
        - `task_labels`: Union[Discrete, Sparse[Discrete]]
           The task labels for each sample. When task labels are not available,
           the task labels space is Sparse, and entries will be `None`. 
        """
        x_space = base_observation_spaces[self.dataset]
        if not self.transforms:
            # NOTE: When we don't pass any transforms, continuum scenarios still
            # at least use 'to_tensor'.
            x_space = Transforms.to_tensor(x_space)

        # apply the transforms to the observation space.
        for transform in self.transforms:
            x_space = transform(x_space)
        x_space = add_tensor_support(x_space)
        

        task_label_space = spaces.Discrete(self.nb_tasks)
        if not self.task_labels_at_train_time:
            task_label_space = Sparse(task_label_space, 1.0)
        task_label_space = add_tensor_support(task_label_space)

        return NamedTupleSpace(
            x=x_space,
            task_labels=task_label_space,
            dtype=self.Observations,
        )

    @property
    def action_space(self) -> spaces.Discrete:
        """ Action space for this setting. """
        if self.relabel:
            return spaces.Discrete(self.n_classes_per_task)
        return spaces.Discrete(self.num_classes)
        
        # TODO: IDEA: Have the action space only reflect the number of 'current' classes
        # in order to create a "true" class-incremental learning setting.
        n_classes_seen_so_far = 0
        for task_id in range(self.current_task_id):
            n_classes_seen_so_far += self.num_classes_in_task(task_id)
        return spaces.Discrete(n_classes_seen_so_far)


    @property
    def reward_space(self) -> spaces.Discrete:
        return self.action_space

    def apply(self, method: Method, config: Config=None) -> ClassIncrementalResults:
        """Apply the given method on this setting to producing some results."""
        # TODO: It still isn't super clear what should be in charge of creating
        # the config, and how to create it, when it isn't passed explicitly.
        self.config: Config
        if config is not None:
            self.config = config
            logger.debug(f"Using Config {self.config}")
        elif isinstance(getattr(method, "config", None), Config):
            # If the Method has a `config` attribute that is a Config, use that.
            self.config = method.config
            logger.debug(f"Using Config from the Method: {self.config}")
        else:
            logger.debug(f"Parsing the Config from the command-line.")
            self.config = Config.from_args(self._argv, strict=False)
            logger.debug(f"Resulting Config: {self.config}")

        method.configure(setting=self)
        
        # Run the Training loop (which is defined in IncrementalSetting).
        self.train_loop(method)
        # Run the Test loop (which is defined in IncrementalSetting).
        results: ClassIncrementalResults = self.test_loop(method)
        
        logger.info(f"Resulting objective of Test Loop: {results.objective}")
        logger.info(results.summary())
        method.receive_results(self, results=results)
        return results

    def prepare_data(self, data_dir: Path = None, **kwargs):
        self.config = self.config or Config.from_args(self._argv, strict=False)
        
        # if self.batch_size is None:
        #     logger.warning(UserWarning(
        #         f"Using the default batch size of 32. (You can set the "
        #         f"batch size by passing a value to the Setting constructor, or "
        #         f"by setting the attribute inside your 'configure' method) "
        #     ))
        #     self.batch_size = 32
        
        data_dir = data_dir or self.data_dir or self.config.data_dir
        self.make_dataset(data_dir, download=True)
        self.data_dir = data_dir
        super().prepare_data(**kwargs)

    def setup(self, stage: Optional[str] = None, *args, **kwargs):
        """ Creates the datasets for each task.
        TODO: Figure out a way of setting data_dir elsewhere maybe?
        """
        assert self.config
        # self.config = self.config or Config.from_args(self._argv)
        logger.debug(f"data_dir: {self.data_dir}, setup args: {args} kwargs: {kwargs}")
        
        self.train_cl_dataset = self.make_dataset(self.data_dir, download=False, train=True)
        self.test_cl_dataset = self.make_dataset(self.data_dir, download=False, train=False)
        
        self.train_cl_loader: _BaseScenario = self.make_train_cl_loader(self.train_cl_dataset)
        self.test_cl_loader: _BaseScenario = self.make_test_cl_loader(self.test_cl_dataset)

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

    def train_dataloader(self, batch_size: int = None, num_workers: int = None) -> PassiveEnvironment:
        """Returns a DataLoader for the train dataset of the current task. """
        if not self.has_prepared_data:
            self.prepare_data()
        if not self.has_setup_fit:
            self.setup("fit")

        # TODO: Clean this up: decide where num_workers should be stored.
        batch_size = batch_size or self.batch_size
        num_workers = num_workers or self.num_workers

        dataset = self.train_datasets[self.current_task_id]
        # TODO: Add some kind of Wrapper around the dataset to make it
        # semi-supervised.
        env = PassiveEnvironment(
            dataset,
            split_batch_fn=self.split_batch_function(training=True),
            observation_space=self.observation_space,
            action_space=self.action_space,
            reward_space=self.reward_space,
            pin_memory=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        
        if self.config.render:
            # TODO: Add a callback wrapper that calls 'env.render' at each step?
            env = env
            
        if self.train_transforms:
            env = TransformObservation(env, f=self.train_transforms)
        if self.train_env:
            self.train_env.close()
        self.train_env = env
        return self.train_env

    def val_dataloader(self, batch_size: int = None, num_workers: int = None) -> PassiveEnvironment:
        """Returns a DataLoader for the validation dataset of the current task.
        """
        if not self.has_prepared_data:
            self.prepare_data()
        if not self.has_setup_fit:
            self.setup("fit")

        dataset = self.val_datasets[self.current_task_id]
        batch_size = batch_size or self.batch_size
        num_workers = num_workers or self.num_workers
        env = PassiveEnvironment(
            dataset,
            split_batch_fn=self.split_batch_function(training=True),
            observation_space=self.observation_space,
            action_space=self.action_space,
            reward_space=self.reward_space,
            pin_memory=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        if self.val_transforms:
            env = TransformObservation(env, f=self.val_transforms)

        if self.val_env:
            self.val_env.close()
            del self.val_env
        self.val_env = env
        return self.val_env

    def test_dataloader(self, batch_size: int = None, num_workers: int = None) -> PassiveEnvironment["ClassIncrementalSetting.Observations", Actions, Rewards]:
        """Returns a DataLoader for the test dataset of the current task.
        """
        if not self.has_prepared_data:
            self.prepare_data()
        if not self.has_setup_test:
            self.setup("test")

        # Testing this out, we're gonna have a "test schedule" like this to try
        # to imitate the MultiTaskEnvironment in RL. 
        transition_steps = [0] + list(itertools.accumulate(map(len, self.test_datasets)))[:-1]
        # Join all the test datasets.        
        dataset = ConcatDataset(self.test_datasets)
        batch_size = batch_size or self.batch_size
        num_workers = num_workers or self.num_workers
        
        env = PassiveEnvironment(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            split_batch_fn=self.split_batch_function(training=False),
            observation_space=self.observation_space,
            action_space=self.action_space,
            reward_space=self.reward_space,
            pretend_to_be_active=True,
        )
        if self.test_transforms:
            env = TransformObservation(env, f=self.test_transforms)

        # NOTE: Two ways of removing the task labels: Either using a different
        # 'split_batch_fn' at train and test time, or by using this wrapper
        # which is also used in the RL side of the tree:
        # TODO: Maybe remove/simplify the 'split_batch_function'.
        from sequoia.settings.active.continual.wrappers import HideTaskLabelsWrapper
        if not self.task_labels_at_test_time:
            env = HideTaskLabelsWrapper(env)

        self.test_task_schedule = dict.fromkeys(
            [step // (env.batch_size or 1) for step in transition_steps],
            range(len(transition_steps)),
        )
        # a bit hacky, but it works.
        env.task_schedule = self.test_task_schedule
        # TODO: Would this mislead the Method into not observing/getting the last batch ?
        env.max_steps = self.max_steps = len(dataset) // (env.batch_size or 1)

        # TODO: Configure the 'monitoring' dir properly.
        test_dir = "results"
        test_loop_max_steps = len(dataset) // (env.batch_size or 1)
        # TODO: Fix this: iteration doesn't ever end for some reason.

        test_env = ClassIncrementalTestEnvironment(
            env,
            directory=test_dir,
            step_limit=test_loop_max_steps,
            force=True,
            config=self.config,
        )

        if self.test_env:
            self.test_env.close()
        self.test_env = test_env
        return self.test_env

    def split_batch_function(self, training: bool) -> Callable[[Tuple[Tensor, ...]], Tuple[Observations, Rewards]]:
        """ Returns a callable that is used to split a batch into observations and rewards.
        """
        task_classes = {
            i: self.task_classes(i, train=training)
            for i in range(self.nb_tasks)
        }

        def split_batch(batch: Tuple[Tensor, ...]) -> Tuple[Observations, Rewards]:
            """Splits the batch into a tuple of Observations and Rewards.

            Parameters
            ----------
            batch : Tuple[Tensor, ...]
                A batch of data coming from the dataset.

            Returns
            -------
            Tuple[Observations, Rewards]
                A tuple of Observations and Rewards.
            """
            # In this context (class_incremental), we will always have 3 items per
            # batch, because we use the ClassIncremental scenario from Continuum.
            assert len(batch) == 3
            x, y, t = batch

            # Relabel y so it is always in [0, n_classes_per_task) for each task.
            if self.relabel:
                y = relabel(y, task_classes)

            if ((training and not self.task_labels_at_train_time) or 
                (not training and not self.task_labels_at_test_time)):
                # Remove the task labels if we're not currently allowed to have
                # them.
                # TODO: Using None might cause some issues. Maybe set -1 instead?
                t = None

            observations = self.Observations(x=x, task_labels=t)
            rewards = self.Rewards(y=y)
            
            return observations, rewards
        return split_batch
    
    def make_train_cl_loader(self, train_dataset: _ContinuumDataset) -> _BaseScenario:
        """ Creates a train ClassIncremental object from continuum. """
        return ClassIncremental(
            train_dataset,
            nb_tasks=self.nb_tasks,
            increment=self.increment,
            initial_increment=self.initial_increment,
            class_order=self.class_order,
            transformations=self.transforms,
        )

    def make_test_cl_loader(self, test_dataset: _ContinuumDataset) -> _BaseScenario:
        """ Creates a test ClassIncremental object from continuum. """
        return ClassIncremental(
            test_dataset,
            nb_tasks=self.nb_tasks,
            increment=self.test_increment,
            initial_increment=self.test_initial_increment,
            class_order=self.test_class_order,
            transformations=self.transforms,
        )

    def make_dataset(self,
                     data_dir: Path,
                     download: bool = True,
                     train: bool = True,
                     **kwargs) -> _ContinuumDataset:
        # TODO: #7 Use this method here to fix the errors that happen when
        # trying to create every single dataset from continuum.
        data_dir = Path(data_dir)
        if not data_dir.exists():
            data_dir.mkdir(parents=True, exist_ok=True)
        
        if self.dataset in self.available_datasets:
            dataset_class = self.available_datasets[self.dataset]   
            return dataset_class(
                data_path=data_dir,
                download=download,
                train=train,
                **kwargs
            )

        elif self.dataset in self.available_datasets.values():
            dataset_class = self.dataset
            return dataset_class(
                data_path=data_dir,
                download=download,
                train=train,
                **kwargs
            )

        elif isinstance(self.dataset, Dataset):
            logger.info(f"Using a custom dataset {self.dataset}")
            return self.dataset

        else:
            raise NotImplementedError

    # These methods below are used by the MultiHeadModel, mostly when
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

    def num_classes_in_current_task(self, train: bool=None) -> int:
        """ Returns the number of classes in the current task. """
        # TODO: Its ugly to have the 'method' tell us if we're currently in
        # train/eval/test, no? Maybe just make a method for each?     
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
    
    def _check_environments(self):
        """ Do a quick check to make sure that the dataloaders give back the
        right observations / reward types.
        """
        for loader_method in [self.train_dataloader, self.val_dataloader, self.test_dataloader]:
            logger.debug(f"Checking loader method {loader_method.__name__}")
            env = loader_method(batch_size=5)
            obs = env.reset()
            assert isinstance(obs, self.Observations)
            # Convert the observation to numpy arrays, to make it easier to
            # check if the elements are in the spaces.
            obs = obs.numpy()
            # take a slice of the first batch, to get sample tensors.
            first_obs = obs[:, 0]
            # TODO: Here we'd like to be able to check that the first observation
            # is inside the observation space, but we can't do that because the
            # task label might be None, and so that would make it fail.
            x, task_label = first_obs
            if task_label is None:
                assert x in self.observation_space[0]
            else:
                pass # FIXME: 
                # assert first_obs.values() in self.observation_space, (first_obs[0].shape, self.observation_space)

            for i in range(5):
                actions = env.action_space.sample()
                observations, rewards, done, info = env.step(actions)
                assert isinstance(observations, self.Observations), type(observations)
                assert isinstance(rewards, self.Rewards), type(rewards)
                batch_size = observations.batch_size
                actions = env.action_space.sample()
                if done:
                    observations = env.reset()
            env.close()


def relabel(y: Tensor, task_classes: Dict[int, List[int]]) -> Tensor:
    """ Relabel the elements of 'y' to their  index in the list of classes for
    their task.
    
    Example:
    
    >>> import torch
    >>> y = torch.as_tensor([2, 3, 2, 3, 2, 2])
    >>> task_classes = {0: [0, 1], 1: [2, 3]}
    >>> relabel(y, task_classes)
    tensor([0, 1, 0, 1, 0, 0])
    """
    # TODO: Double-check that this never leaves any zeros where it shouldn't.
    new_y = torch.zeros_like(y)
    unique_y = set(torch.unique(y).tolist())
    # assert unique_y <= set(task_classes), (unique_y, task_classes)
    for task_id, task_true_classes in task_classes.items():
        for i, label in enumerate(task_true_classes):
            new_y[y == label] = i
    return new_y

# This is just meant as a cleaner way to import the Observations/Actions/Rewards
# than particular setting.
Observations = ClassIncrementalSetting.Observations
Actions = ClassIncrementalSetting.Actions
Rewards = ClassIncrementalSetting.Rewards

# TODO: I wouldn't want these above to overwrite / interfere with the import of
# the "base" versions of these objects from sequoia.settings.bases.objects, which are
# imported in settings/__init__.py. Will have to check that doing
# `from .passive import *` over there doesn't actually import these here.


class ClassIncrementalTestEnvironment(TestEnvironment):
    def __init__(self, env: gym.Env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self._steps = 0
        self.task_steps = sorted(self.env.task_schedule.keys())
        self.metrics: List[ClassificationMetrics] = [[] for step in self.task_steps]
        self._reset = False

    def get_results(self) -> ClassIncrementalResults:
        rewards = self.get_episode_rewards()
        lengths = self.get_episode_lengths()
        total_steps = self.get_total_steps()
        n_metrics_per_task = [len(task_metrics) for task_metrics in self.metrics]
        return ClassIncrementalResults(
            test_metrics=self.metrics
        )

    def reset(self):
        if not self._reset:
            logger.debug(f"Initial reset.")
            self._reset = True
            return super().reset()
        else:
            logger.debug(f"Resetting the env closes it.")
            self.close()
            return None
        
    def _before_step(self, action):
        self.action = action
        return super()._before_step(action)
    
    def _after_step(self, observation, reward, done, info):
        
        assert isinstance(reward, Tensor)
        action = self.action
        actions = torch.as_tensor(action)
        
        batch_size = reward.shape[0]
        fake_logits = torch.zeros([batch_size, self.action_space.nvec[0]], dtype=int)
        # FIXME: There must be a smarter way to do this indexing.
        for i, action in enumerate(actions):
            fake_logits[i, action] = 1
        actions = fake_logits
        
        metric = ClassificationMetrics(y=reward, y_pred=actions)
        reward = metric.accuracy
        
        task_steps = sorted(self.task_schedule.keys())
        assert 0 in task_steps, task_steps
        import bisect
        nb_tasks = len(task_steps)
        assert nb_tasks >= 1

        import bisect
        # Given the step, find the task id.
        task_id = bisect.bisect_right(task_steps, self._steps) - 1
        self.metrics[task_id].append(metric)
        self._steps += 1
        
        
        ## Debugging issue with Monitor class:
        # return super()._after_step(observation, reward, done, info)
        if not self.enabled: return done

        if done and self.env_semantics_autoreset:
            # For envs with BlockingReset wrapping VNCEnv, this observation will be the first one of the new episode
            if self.config.render:
                self.reset_video_recorder()
            self.episode_id += 1
            self._flush()

        # Record stats
        self.stats_recorder.after_step(observation, reward, done, info)
        
        # Record video
        if self.config.render:
            self.video_recorder.capture_frame()
        return done
        ## 
    
    
    def _after_reset(self, observation: ClassIncrementalSetting.Observations):
        image_batch = observation.numpy().x
        # Need to create a single image with the right dtype for the Monitor
        # from gym to create gifs / videos with it.
        if self.batch_size:
            # Need to tile the image batch so it can be seen as a single image
            # by the Monitor.
            image_batch = tile_images(image_batch)

        image_batch = Transforms.channels_last_if_needed(image_batch)
        if image_batch.dtype == np.float32:
            assert (0 <= image_batch).all() and (image_batch <= 1).all()
            image_batch = (256 * image_batch).astype(np.uint8)
        
        assert image_batch.dtype == np.uint8
        # Debugging this issue here:
        # super()._after_reset(image_batch)

        ## -- Code from Monitor
        if not self.enabled: return
        # Reset the stat count
        self.stats_recorder.after_reset(observation)
        if self.config.render:
            self.reset_video_recorder()

        # Bump *after* all reset activity has finished
        self.episode_id += 1

        self._flush()
        ## -- 

    def render(self, mode='human', **kwargs):
        # NOTE: This doesn't get called, because the video recorder uses
        # self.env.render(), rather than self.render()
        # TODO: Render when the 'render' argument in config is set to True.        
        image_batch = super().render(mode=mode, **kwargs)
        if mode == "rgb_array" and self.batch_size:
            image_batch = tile_images(image_batch)
        return image_batch

if __name__ == "__main__":
    import doctest
    doctest.testmod()
