""" Defines a `Setting` subclass for "Class-Incremental" Continual Learning.

Example command to run a method on this setting (in debug mode):
```
python main.py --setting class_incremental --method baseline --debug  \
    --batch_size 10 --max_epochs 1
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
import warnings
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import (Callable, ClassVar, Dict, List, Optional, Tuple, Type, Union)

import matplotlib.pyplot as plt
import torch
from continuum import ClassIncremental
from continuum.tasks import split_train_val
from continuum.datasets import *
from continuum.datasets import _ContinuumDataset
from continuum.scenarios.base import _BaseCLLoader
from pytorch_lightning import LightningModule, Trainer
from simple_parsing import choice, list_field
from torch import Tensor
from tqdm import tqdm
from torch.utils.data import DataLoader
        
from common import ClassificationMetrics, Metrics, get_metrics
from common.config import Config
from common.loss import Loss
from common.transforms import Transforms
from settings.base import Results
from settings.base.environment import ObservationType, RewardType
from utils import dict_union, get_logger
from utils.utils import constant, mean

from ..passive_setting import PassiveSetting
from ..passive_environment import PassiveEnvironment
from .class_incremental_results import ClassIncrementalResults

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
class ClassIncrementalSetting(PassiveSetting[ObservationType, RewardType]):
    """Settings where the data is non-stationary, and grouped into tasks.

    The current task can be set at the `current_task_id` attribute.
    """
    # Class variable holding a dict of the names and types of all available
    # datasets.
    available_datasets: ClassVar[Dict[str, Type[_ContinuumDataset]]] = {
        c.__name__.lower(): c
        for c in [
            CIFARFellowship, Fellowship, MNISTFellowship,
            ImageNet100, ImageNet1000,
            MultiNLI,
            CIFAR10, CIFAR100, EMNIST, KMNIST, MNIST, QMNIST, FashionMNIST,
        ]
    }
    # A continual dataset to use. (Should be taken from the continuum package).
    dataset: str = choice(available_datasets.keys(), default="mnist")
    # Default path to which the datasets will be downloaded.
    data_dir: Path = Path("data")

    # Transformations to use. See the Transforms enum for the available values.
    transforms: List[Transforms] = list_field(
        Transforms.to_tensor,
        # BUG: The input_shape given to the Model doesn't have the right number
        # of channels, even if we 'fixed' them here. However the images are fine
        # after.
        Transforms.fix_channels,
        Transforms.channels_first_if_needed,
    )
    
    # Wether task labels are available at train time.
    # NOTE: Forced to True at the moment.
    task_labels_at_train_time: bool = constant(True)
    # Wether task labels are available at test time.
    task_labels_at_test_time: bool = False
    # Wether we get informed when reaching the boundary between two tasks during
    # training. Only used when `smooth_task_boundaries` is False.
    # TODO: Setting constant for now, but we could add task boundary detection
    # later on!
    known_task_boundaries_at_train_time: bool = constant(True)
    # Wether we get informed when reaching the boundary between two tasks during
    # training. Only used when `smooth_task_boundaries` is False.
    known_task_boundaries_at_test_time: bool = constant(True)

    # TODO: Actually add the 'smooth' task boundary case.
    # Wether we have clear boundaries between tasks, or if the transition is
    # smooth.
    smooth_task_boundaries: bool = constant(False) # constant for now.

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
       
        super().__post_init__(
            obs_shape=image_shape,
            action_shape=self.n_classes_per_task,
            reward_shape=1, # the labels have shape (1,) always.
        )
        self._current_task_id: int = 0
        self.train_datasets: List[_ContinuumDataset] = []
        self.val_datasets: List[_ContinuumDataset] = []
        self.test_datasets: List[_ContinuumDataset] = []
        
        # This will be set by the Experiment, or passed to the `apply` method.
        # TODO: This could be a bit cleaner.
        self.config: Config = None

    def apply(self, method: "Method", config: Config) -> ClassIncrementalResults:
        """Apply the given method on this setting, producing some results.

        The pseudocode for this evaluation procedure looks a bit like this:

        ```python
        # training loop:

        for i in range(self.nb_tasks):
            self.current_task_id = i
            
            # Inform the model of a task boundary. If the task labels are
            # available, then also let the method know the index of the new
            # task. 
            if self.known_task_boundaries_at_train_time:
                if not self.task_labels_at_train_time:
                    method.on_task_switch(None)
                else:
                    method.on_task_switch(task_id=i)

            # Train the method using train/val datasets of task i:
            method.fit(
                train_dataloader=self.train_dataloader(self.train_datasets[i]),
                val_dataloader=self.val_dataloader(self.train_datasets[i]),
            )

        task_accuracies: List[float] = []
        for i in range(self.nb_tasks):
            self.current_task_id = i

            # Same as above, but for testing:
            if self.known_task_boundaries_at_test_time:
                if not self.task_labels_at_test_time:
                    method.on_task_switch(None)
                else:
                    method.on_task_switch(task_id=i)
        
            test_i_dataloader = self.test_dataloader(self.test_datasets[i])

            total: int = 0
            correct: int = 0

            # Manually loop over the test dataloader of task i:
            for (x, y, task_labels) in test_i_dataloader:
                if not self.task_labels_at_test_time:
                    task_labels = None

                y_pred = method.predict(x=x, y=y, task_labels=task_labels)
                total += len(x)
                correct += sum(np.argmax(y_pred, -1) == y)
            
            task_accuracy = correct / total
            task_accuracies.append(task_accuracy)
        
        return sum(task_accuracies) / len(task_accuracies)
        ```

        :param method: [description]
        :type method: Method
        :param config: [description]
        :type config: Config
        :raises RuntimeError: [description]
        :return: [description]
        :rtype: [type]
        """
        # NOTE: (@lebrice) The test loop is written by hand here because I don't
        # want to have to give the labels to the method at test-time. See the
        # docstring of `test_loop` for more info.

        # TODO: (@lebrice) Currently redesigning this. There were a few problems
        # with the previous approach: giving back the loaders for all the tasks at
        # once in `val_dataloader` and in `test_dataloader`, as is natural with
        # the `LightningDataModule` API, indirectly gives the task ids to the
        # method through the `dataloader_idx` argument that gets passed to the
        # `val_step` and `test_step` methods of the LightningModule. We instead do
        # something a bit more flexible, which is to allow settings to specify
        # the train/test loop themselves completely, while still mimicking the
        # Trainer API on the Method, just so if users (methods) want to use a
        # Trainer and a LightningModule, they are free to do so.
        from methods import Method
        method: Method
        
        # TODO: At the moment, we're nice enough to do this, but this would
        # maybe allow the method to "cheat"!
        method.config = config
        method.configure(setting=self)


        self.config = config
        # Get the arguments that will be used to create the dataloaders.
        # TODO: We create the dataloaders here and pass them to the method, but
        # we could maybe do this differently. This isn't super clean atm.
        # Get the batch size from the model, or the config, or 32.
        batch_size = getattr(method, "batch_size", getattr(config, "batch_size", 32))
        # Get the data dir from self, the config, or 'data'.
        data_dir = getattr(self, "data_dir", getattr(config, "data_dir", "data"))
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
        
        # Training loop:
        for task_id in range(self.nb_tasks):
            logger.info(f"Starting training on task {task_id}")
            # Update the task id internally.
            self.current_task_id = task_id

            assert not self.smooth_task_boundaries, "TODO: (#18) Make another 'Continual' setting that supports smooth task boundaries."
            
            if self.known_task_boundaries_at_train_time:
                # Inform the model of a task boundary. If the task labels are
                # available, then also give the id of the new task to the
                # method.
                # TODO: Should we also inform the method of wether or not the
                # task switch is occuring during training or testing?
                if not self.task_labels_at_train_time:
                    method.on_task_switch(None)
                else:
                    method.on_task_switch(task_id)
            
            # Starting with something super simple, creating the dataloaders
            # ourselves (rather than passing 'self' as the datamodule):
            # success = trainer.fit(model, datamodule=self)
            task_train_loader = self.train_dataloader(**dataloader_kwargs)
            task_val_loader = self.val_dataloader(**dataloader_kwargs)
            
            success = method.fit(
                train_dataloader=task_train_loader,
                val_dataloaders=task_val_loader,
                # datamodule=self,
            )
            if success:
                logger.debug(f"Finished Training on task {task_id}.")
            if not success:
                raise RuntimeError(
                    f"Something didn't work during training: "
                    f"method.fit() returned {success}"
                )

        results: ClassIncrementalResults = self.test_loop(method)
        logger.info(f"Resulting objective of Test Loop: {results.objective}")
        print(results.summary())
        print(results.to_log_dict())
        results.save_to_dir(self.config.log_dir)
        return results

    def test_loop(self, method: "Method") -> ClassIncrementalResults:
        """ (WIP): Runs the class-incremental CL test loop and returns the Results.

        Args:
            method (Method): The Method to evaluate.

        Returns:
            `ClassIncrementalResults`:  An object that holds the test metrics
            and that is used to calculate the `objective`, a float representing
            how 'good' this method was on this given setting. This object is
            also useful for creating the plots, serialization, and logging to
            wandb. See `ClassIncrementalResults` for more info.

        Important Notes:
        - **Not all settings need to have a `test_loop` method!** The 'training'
            and 'testing' logic could all be mixed together in the `apply`
            method in whatever way you want! (e.g. test-time training, OSAKA,
            etc.) This method is just here to make the `apply` method a bit more
            tidy.
        
        -   The PL way of doing this here would be something like:
            `test_results = method.test(datamodule=self)`, however, there are
            some issues with doing it this way (as I just recently learned):
            - This gives the method/model access to the labels at test time;
            - The Method/LightningModule gets the responsibility of creating the
              metrics we're interested in measuring in its `test_step` method.
            - It might be difficult to customize the test loop. For example,
              How would one go about adding some kind of 'test-time training'
              or OSAKA-like evaluation setup using the usual
              `[train/val/test]_step` methods?

            However, I'd rather not do that, and write out the test loop
            manually here, which also allows us more flexibility, but also has
            some downsides:
            - We don't currently support any of the Callbacks from
              pytorch-lightning during testing.
              
            For some subclasses (e.g `IIDSetting`), it might be totally fine to
            just use the usual Trainer.fit() and Trainer.test() methods, so feel
            free to overwrite this method with your own implementation if that
            makes your life easier.
        """
        from methods import Method
        method: Method

        # Create a list that will hold the test metrics encountered during
        test_metrics: List[List[Metrics]] = [
            [] for _ in range(self.nb_tasks)
        ]

        for task_id in range(self.nb_tasks):
            self.current_task_id = task_id

            assert not self.smooth_task_boundaries, "TODO: (#18) Make another 'Continual' setting that supports smooth task boundaries."
            # Inform the model of a task boundary. If the task labels are
            # available, then also give the id of the new task.
            # TODO: Should we also inform the method of wether or not the task
            # switch is occuring during training or testing?
            if self.known_task_boundaries_at_test_time:
                if not self.task_labels_at_test_time:
                    method.on_task_switch(None)
                else:
                    method.on_task_switch(task_id)

            # Manual test loop:
            test_task_loader = self.test_dataloader(**self.dataloader_kwargs)
            pbar = tqdm(test_task_loader)
            for x_batch, y_batch, task_labels in pbar:
                assert not any(y_batch >= self.n_classes_per_task), (y_batch, self.n_classes_per_task)
                if not self.task_labels_at_test_time:
                    # If the task labels aren't given at test time, then we
                    # give a list of Nones. We could also maybe give a portion
                    # of the task labels, which could be interesting.
                    task_labels = [None] * len(x_batch)
                # Get the predicted label for this batch of inputs.
                y_pred = method.predict(x=x_batch, task_labels=task_labels)
                # Get the metrics for that batch.
                batch_metrics = self.get_metrics(y_pred=y_pred, y=y_batch)
                # TODO: Remove this, just debugging.
                assert isinstance(batch_metrics, ClassificationMetrics)
                # Save the metrics for this batch in the list above.
                test_metrics[task_id].append(batch_metrics)
                pbar.set_postfix(batch_metrics.to_pbar_message())
            pbar.close()
            
            average_task_metrics = mean(test_metrics[task_id])
            logger.info(f"Results on task {task_id}: {average_task_metrics}")

        results = ClassIncrementalResults(test_metrics=test_metrics)
        return results

    def get_metrics(self,
                    y_pred: Tensor,
                    y: Tensor) -> Union[float, Metrics]:
        """ Calculate the "metric" from the model prediction and the true label.
        
        In this example, we return a 'Metrics' object:
        - `ClassificationMetrics` for classification problems,
        - `RegressionMetrics` for regression problems.
        
        We use these objects because they are awesome (they basically simplify
        making plots, wandb logging, and serialization), but you can also just
        return floats if you want, no problem.
        """
        # TODO: For unsupervised / semi-supervised training, we could accept
        # something else here, no?
        from common.metrics import get_metrics
        y_pred = y_pred.cpu().detach()
        y = y.cpu().detach()
        return get_metrics(y_pred=y_pred, y=y)

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

    # LightningDataModule methods:
    
    def prepare_data(self, data_dir: Path = None, **kwargs):
        data_dir = data_dir or self.data_dir
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
        """Returns a DataLoader for the train dataset of the current task.
        
        NOTE: The dataloader is passive for now (just a regular DataLoader).
        """
        if not self.has_prepared_data:
            self.prepare_data()
        if not self.has_setup_fit:
            self.setup("fit")
        dataset = self.train_datasets[self.current_task_id]
        # TODO: Add some kind of Wrapper around the dataset to make the dataset
        # semi-supervised.
        kwargs = dict_union(self.dataloader_kwargs, kwargs)
        batch_transforms = [
            RelabelTransform(task_classes=self.current_task_classes(train=True))
        ]
        env: DataLoader = PassiveEnvironment(
            dataset,
            batch_transforms=batch_transforms,
            **kwargs,
        )
        # TODO: Add some kind of wrapper that hides the task labels from
        # continuum during Training.
        return env

    def val_dataloader(self, **kwargs) -> PassiveEnvironment:
        """Returns a DataLoader for the validation dataset of the current task.
        
        NOTE: The dataloader is passive for now (just a regular DataLoader).
        """
        if not self.has_prepared_data:
            self.prepare_data()
        if not self.has_setup_fit:
            self.setup("fit")
        dataset = self.val_datasets[self.current_task_id]
        kwargs = dict_union(self.dataloader_kwargs, kwargs)
        batch_transforms = [
            RelabelTransform(task_classes=self.current_task_classes(train=True))
        ]
        env: DataLoader = PassiveEnvironment(
            dataset,
            batch_transforms=batch_transforms,
            **kwargs,
        )
        return env

    def test_dataloader(self, **kwargs) -> PassiveEnvironment:
        """Returns a DataLoader for the test dataset of the current task.

        NOTE: The dataloader is passive for now (just a regular DataLoader).
        """
        if not self.has_prepared_data:
            self.prepare_data()
        if not self.has_setup_test:
            self.setup("test")
        dataset = self.test_datasets[self.current_task_id]
        kwargs = dict_union(self.dataloader_kwargs, kwargs)
        batch_transforms = [
            RelabelTransform(task_classes=self.current_task_classes(train=False))
        ]
        env: DataLoader = PassiveEnvironment(
            dataset,
            batch_transforms=batch_transforms,
            **kwargs,
        )
        return env

    @property
    def current_task_id(self) -> Optional[int]:
        """ Get the current task id.
        
        TODO: Do we want to return None if the task labels aren't currently
        available? (at either Train or Test time?) We'd then have to detect if
        we're training or testing from within the Setting.. Could maybe use the
        setup() method to set some property or something like that?
        """
        return self._current_task_id

    @current_task_id.setter
    def current_task_id(self, value: int) -> None:
        """ Sets the current task id. """
        self._current_task_id = value

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


@dataclass
class RelabelTransform(Callable[[Tuple[Tensor, ...]], Tuple[Tensor, ...]]):
    """ Transform that puts labels back into the [0, n_classes_per_task] range.
    
    For instance, if it's given a bunch of images that have labels [2, 3, 2]
    and the `task_classes = [2, 3]`, then the new labels will be
    `[0, 1, 0]`.
    
    Note that the order in `task_classes` is perserved. For instance, in the
    above example, if `task_classes = [3, 2]`, then the new labels would be
    `[1, 0, 1]`.
    """
    task_classes: List[int] = list_field()
    
    def __call__(self, batch: Tuple[Tensor, ...]):
        if isinstance(batch, list):
            batch = tuple(batch)
        if not isinstance(batch, tuple):
            return batch
        if len(batch) == 1:
            return batch        
        x, y, *extra_inputs = batch
        new_y = torch.empty_like(y)
        for i, label in enumerate(self.task_classes):
            new_y[y == label] = i
        return (x, new_y, *extra_inputs)
