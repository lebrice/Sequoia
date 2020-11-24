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
import itertools
import dataclasses
import warnings
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import (Callable, ClassVar, Dict, List, Optional, Sequence, Tuple,
                    Type, Union)

import gym
import matplotlib.pyplot as plt
import torch
import numpy as np
from continuum import ClassIncremental
from continuum.datasets import *
from continuum.datasets import _ContinuumDataset
from continuum.scenarios.base import _BaseCLLoader
from continuum.tasks import split_train_val
from gym import spaces
from pytorch_lightning import LightningModule, Trainer
from simple_parsing import choice, list_field
from torch import Tensor
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from common import ClassificationMetrics, Metrics, get_metrics
from common.config import Config
from common.loss import Loss
from common.spaces import Sparse
from common.transforms import Transforms, SplitBatch, Compose
from settings.base import Method, Results, ObservationType, RewardType
from utils import dict_union, get_logger, constant, mean, take

from .class_incremental_results import ClassIncrementalResults
from ..passive_setting import PassiveSetting
from ..passive_environment import PassiveEnvironment, Actions, ActionType, Observations, Rewards

from settings.assumptions.incremental import IncrementalSetting, TestEnvironment

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
}

dims_for_dataset: Dict[str, Tuple[int, int, int]] = {
    "mnist": (28, 28, 1),
    "fashionmnist": (28, 28, 1),
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
            CIFARFellowship, MNISTFellowship, ImageNet100,
            ImageNet1000, CIFAR10, CIFAR100, EMNIST, KMNIST, MNIST,
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
        if not hasattr(self, "num_classes") and not hasattr(self.dataset, "num_classes"):
            # In some concrete LightningDataModule's like MnistDataModule,
            # num_classes is a read-only property. Therefore we check if it
            # is already defined. This is just in case something tries to
            # inherit from both IIDSetting and MnistDataModule, for instance.
            if self.dataset not in num_classes_in_dataset:
                self.dataset = self.dataset.lower().replace("_", "")
            if self.dataset not in num_classes_in_dataset:
                raise NotImplementedError(
                    f"Can't tell how many classes there are in dataset "
                    f"{self.dataset}, as it isn't in the "
                    f"num_classes_in_dataset dict, and doesn't have a "
                    f"'num_classes' attribute. (num_classes_in_dataset "
                    f"keys: {num_classes_in_dataset.keys()}"
                )
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
        image_space = spaces.Box(low=0, high=1, shape=image_shape, dtype=np.float32)
        task_label_space = spaces.Discrete(self.nb_tasks)
        if not self.task_labels_at_train_time:
            task_label_space = Sparse(task_label_space, 1.0)
        observation_space = spaces.Tuple([
            image_space,
            task_label_space,
        ])
        # assert False, image_space
        # TODO: Change the actions from logits to predicted labels.
        action_space = spaces.Discrete(self.n_classes_per_task)
        # self.action_space = Box(low=-np.inf, high=np.inf, shape=(self.n_classes_per_task,))
        # self.reward_space = spaces.Discrete(self.num_classes)
        reward_space = spaces.Discrete(self.n_classes_per_task)

        super().__post_init__(
            observation_space=observation_space,
            action_space=action_space,
            reward_space=reward_space, # the labels have shape (1,) always.
        )
        image_space = self.train_transforms.space_change(image_space)
        self.observation_space = spaces.Tuple([
            image_space,
            task_label_space,
        ])
        

        self.train_datasets: List[_ContinuumDataset] = []
        self.val_datasets: List[_ContinuumDataset] = []
        self.test_datasets: List[_ContinuumDataset] = []
        
        # This will be set by the Experiment, or passed to the `apply` method.
        # TODO: This could be a bit cleaner.
        self.config: Config
        # Default path to which the datasets will be downloaded.
        self.data_dir: Optional[Path] = None

    def apply(self, method: Method, config: Config=None) -> ClassIncrementalResults:
        """Apply the given method on this setting to producing some results."""
        self.config = config or Config.from_args(self._argv)
        method.config = self.config

        self.configure(method)
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
        self.config = self.config or Config.from_args(self._argv)
        
        if self.batch_size is None:
            logger.warning(UserWarning(
                f"Using the default batch size of 32. (You can set the "
                f"batch size attribute of the setting inside your 'configure' "
                f"method) "
            ))
            self.batch_size = 32
        
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

    def train_dataloader(self, batch_size: int = None, num_workers: int = None) -> PassiveEnvironment:
        """Returns a DataLoader for the train dataset of the current task. """
        if not self.has_prepared_data:
            self.prepare_data()
        if not self.has_setup_fit:
            self.setup("fit")
        batch_size = batch_size or self.batch_size
        num_workers = num_workers or self.config.num_workers
        dataset = self.train_datasets[self.current_task_id]
        # TODO: Add some kind of Wrapper around the dataset to make it
        # semi-supervised.        
        self.train_env = PassiveEnvironment(
            dataset,
            split_batch_fn=self.split_batch_function(training=True),
            observation_space=self.observation_space,
            action_space=self.action_space,
            reward_space=self.reward_space,
            pin_memory=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        # TODO: Do we want to update self.observation_space here?
        # We want to keep the spaces 'un-batched', so we keep a slice across the
        # first dimension.
        # self.observation_space = self.train_env.observation_space[0]
        # self.action_space = self.train_env.action_space[0]
        # self.reward_space = self.train_env.reward_space[0]
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

        # batch_transforms: List[Callable] = self.val_batch_transforms()
        self.val_env = PassiveEnvironment(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            split_batch_fn=self.split_batch_function(training=True),
            observation_space=self.observation_space,
            action_space=self.action_space,
            reward_space=self.reward_space,
        )
        return self.val_env

    def test_dataloader(self, batch_size: int = None, num_workers: int = None) -> PassiveEnvironment["ClassIncrementalSetting.Observations", Actions, Rewards]:
        """Returns a DataLoader for the test dataset of the current task.
        """
        if not self.has_prepared_data:
            self.prepare_data()
        if not self.has_setup_test:
            self.setup("test")

        # Testing this out, we're gonna have a "test schedule" like this!
        transition_steps = [0] + list(itertools.accumulate(map(len, self.test_datasets)))[:-1]
        
        dataset = ConcatDataset(self.test_datasets)
        batch_size = batch_size or self.batch_size
        assert batch_size, "batch size is None?"
        num_workers = num_workers or self.num_workers
        
        # batch_transforms: List[Callable] = self.test_batch_transforms()
        # FIXME: the transform that splits the batch is actually changing the
        # shape of the observation space of the environment!

        dataloader = PassiveEnvironment(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            split_batch_fn=self.split_batch_function(training=False),
            observation_space=self.observation_space,
            action_space=self.action_space,
            reward_space=self.reward_space,
        )
        self.test_task_schedule = dict.fromkeys(
            [step // (dataloader.batch_size or 1) for step in transition_steps],
            range(len(transition_steps)),
        )
        # a bit hacky, but it works.
        dataloader.task_schedule = self.test_task_schedule
        # TODO: Would this mislead the Method into not observing/getting the last batch ?
        dataloader.max_steps = self.max_steps = len(dataset) // dataloader.batch_size
        
        # TODO: Configure the 'monitoring' dir properly.
        test_dir = "results"
        test_loop_max_steps = len(dataset) // dataloader.batch_size
        self.test_env = ClassIncrementalTestEnvironment(
            dataloader,
            directory=test_dir,
            step_limit=test_loop_max_steps,
            force=True,
        )
        return self.test_env

    def split_batch_function(self, training: bool) -> Callable[[Tuple[Tensor, ...]], Tuple[Observations, Rewards]]:
        """ Returns a callable that can be used to split a batch into observations and rewards.
        """
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
            task_labels_in_batch = torch.unique(t).tolist()

            # Index using the task labels, to get the classes for each
            # task label, and then relabel each y to its index within the labels
            # of its corresponding task.
            # TODO: Make sure that this is how we want to do this. This
            # wouldn't make sense for example if successive tasks could use
            # the same input image, but with a different label!

            # NOTE: This supports relabeling data from multiple tasks.
            
            all_indices = np.arange(x.shape[0])
            for task_label in task_labels_in_batch:
                # Get the set of classes for the task at index `t`.
                classes_in_task = self.task_classes(task_label, train=training)
                # Relabel that portion of the labels.
                indices = all_indices[t == task_label]
                y[indices] = relabel(y[indices], task_classes=classes_in_task)

            # Make sure that the labels are in [0, n_classes_per_task] range:
            assert 0 <= y.min(), y
            assert y.max() < self.n_classes_per_task, y
            
            # Re-arrange tensors: (x, y, t) -> ((x, t), y)
            observations = (x, t)
            rewards = y
            
            if ((training and not self.task_labels_at_train_time) or 
                (not training and not self.task_labels_at_test_time)):
                # Remove the task labels if we're not currently allowed to have
                # them.
                # TODO: Using None might cause some issues. Maybe set -1 instead?
                observations = (x, None)

            # Create the 'Observations' and 'Rewards' objects.
            observations = self.Observations.from_inputs(observations)
            rewards = self.Rewards.from_inputs(rewards)
            return observations, rewards
        return split_batch
    
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
                assert first_obs in self.observation_space
            
            assert len(first_obs) == len(self.size())
            for sample_item, expected_shape in zip(first_obs, self.size()):
                if isinstance(sample_item, (Tensor, np.ndarray)):
                    assert sample_item.shape == expected_shape

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


def relabel(y: Tensor, task_classes: List[int]) -> Tensor:
    """ Relabel the elements of 'y' to their  index in the task classes.
    
    Example:
    
    >>> import torch
    >>> y = torch.as_tensor([2, 3, 2, 3, 2, 2])
    >>> task_classes = [2, 3]
    >>> relabel(y, task_classes)
    tensor([0, 1, 0, 1, 0, 0])
    """
    # TODO: Double-check that this never leaves any zeros where it shouldn't.
    new_y = torch.zeros_like(y)
    unique_y = set(torch.unique(y).tolist())
    assert unique_y <= set(task_classes), (unique_y, task_classes)
    for i, label in enumerate(task_classes):
        new_y[y == label] = i
    return new_y

# This is just meant as a cleaner way to import the Observations/Actions/Rewards
# than particular setting.
Observations = ClassIncrementalSetting.Observations
Actions = ClassIncrementalSetting.Actions
Rewards = ClassIncrementalSetting.Rewards

# TODO: I wouldn't want these above to overwrite / interfere with the import of
# the "base" versions of these objects from settings.bases.objects, which are
# imported in settings/__init__.py. Will have to check that doing
# `from .passive import *` over there doesn't actually import these here.


class ClassIncrementalTestEnvironment(TestEnvironment):
    def __init__(self, env: gym.Env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self._steps = 0
        self.task_steps = sorted(self.env.task_schedule.keys())
        self.metrics: List[ClassificationMetrics] = [[] for step in self.task_steps]
        

    def get_results(self) -> ClassIncrementalResults:
        rewards = self.get_episode_rewards()
        lengths = self.get_episode_lengths()
        total_steps = self.get_total_steps()
        n_metrics_per_task = [len(task_metrics) for task_metrics in self.metrics]
        return ClassIncrementalResults(
            test_metrics=self.metrics
        )
    
    def _before_step(self, action):
        self.action = action
        return super()._before_step(action)
    
    def _after_step(self, observation, reward, done, info):
        
        assert isinstance(reward, Tensor)
        actions = torch.as_tensor(self.action)
        
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
        return super()._after_step(observation, reward, done, info)

    def _after_reset(self, observation: ClassIncrementalSetting.Observations):
        image_batch = observation.numpy().x
        from common.gym_wrappers.batch_env.tile_images import tile_images
        big_image = tile_images(image_batch)
        return super()._after_reset(big_image)

    def render(self, mode='human', **kwargs):
        from common.gym_wrappers.batch_env.tile_images import tile_images
        image_batch = super().render(mode=mode, **kwargs)
        if mode == "rgb_array":
            return tile_images(image_batch)
        return image_batch
        


if __name__ == "__main__":
    import doctest
    doctest.testmod()
