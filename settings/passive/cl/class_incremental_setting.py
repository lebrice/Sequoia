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
from dataclasses import dataclass
from pathlib import Path
from typing import (Callable, ClassVar, Dict, List, Optional, Tuple, Type, Union)
import torch

from continuum import ClassIncremental, split_train_val
from continuum.datasets import *
from continuum.datasets import _ContinuumDataset
from continuum.scenarios.base import _BaseCLLoader
from pytorch_lightning import LightningModule, Trainer
from simple_parsing import choice, list_field
from torch import Tensor
from torch.utils.data import DataLoader
        
from common import ClassificationMetrics, Metrics, get_metrics
from common.config import Config
from common.loss import Loss
from common.transforms import Transforms
from settings.base import Results
from settings.base.environment import ObservationType, RewardType
from utils import dict_union, get_logger
from utils.utils import constant

from ..passive_setting import PassiveSetting
from ..passive_environment import PassiveEnvironment
from .results import ClassIncrementalResults

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
    results_class: ClassVar[Type[Results]] = ClassIncrementalResults

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
    increment: Union[List[int], int] = list_field(2, type=int, nargs="*", alias="n_classes_per_task")
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

        super().__post_init__(
            obs_shape=image_shape,
            action_shape=self.num_classes,
            reward_shape=self.num_classes,
        )

        self._current_task_id: int = 0

        self.train_datasets: List[_ContinuumDataset] = []
        self.val_datasets: List[_ContinuumDataset] = []
        self.test_datasets: List[_ContinuumDataset] = []

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

    def apply(self, method: "Method", config: Config):
        """Apply the given method on this setting.

        The pseudocode for this evaluation procedure looks like this:

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
        model = method.model
        # Get the batch size from the model, or the config, or 32.
        batch_size = getattr(model, "batch_size", getattr(config, "batch_size", 32))
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
        # TODO: This might not necessarily make sense to do manually, if we were
        # in a DDP setup. But for now it should be fine.
        # Download the data, if needed.
        if not self.has_prepared_data:
            self.prepare_data(data_dir=data_dir)
        # Call the 'setup' hook for fit, if needed.
        if not self.has_setup_fit:
            self.setup("fit")

        # Training loop:
        for task_id in range(self.nb_tasks):
            logger.debug(f"Starting (new) training routine on task {task_id}")
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
            logger.debug(f"Sucess: {success}")
            if not success:
                raise RuntimeError(
                    f"Something didn't work during training: "
                    f"method.fit() returned {success}"
                )

        test_accuracy = self.test_loop(method)
        logger.info(f"Results of Test Loop: {test_accuracy}")
        return test_accuracy

    def test_loop(self, method: "Method") -> float:
        """ Runs the "usual" continual learning 'Test loop'.

        Args:
            method (Method): The Method to evaluate.

        Returns:
            Metrics: the 'test loop' performance metrics. This is a `Metrics`
            object at the moment, but it could also just be a float.
            These are used here because they are pretty cool, and just make it
            easier than having to move around some dicts for the metrics. For
            example, when in a Classification task, they also give the confusion
            matrix, the class accuracy, and could be used to make some neat
            wandb plots!
            
        Important Notes:
        - **Not all settings need to have a `test_loop` method!** The 'training'
            and 'testing' logic could all be mixed together in the `apply`
            method in whatever way you want! (e.g. test-time training, OSAKA,
            etc.) This method is just here to make the `apply` method a bit more
            tidy.
        
        -   The PL way of doing this here would be something like:
            `test_results = method.test(datamodule=self)`, however, there are
            some issues with doing it this way (as I have recently learned):
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
        
        if not self.has_prepared_data:
            self.prepare_data()
        if not self.has_setup_test:
            self.setup("test")

        # List of metrics for each task.
        metrics_for_each_task: List[Metrics] = []
        
        # Test loop:
        for task_id in range(self.nb_tasks):
            self.current_task_id = task_id
            
            # Use a `Metrics` object to get more  
            task_metrics: Metrics = Metrics()
                        
            assert not self.smooth_task_boundaries, "TODO: (#18) Make another 'Continual' setting that supports smooth task boundaries."
            if self.known_task_boundaries_at_test_time:
                # Inform the model of a task boundary. If the task labels are
                # available, then also give the id of the new task to the
                # method.
                # TODO: Should we also inform the method of wether or not the
                # task switch is occuring during training or testing?
                if not self.task_labels_at_test_time:
                    method.on_task_switch(None)
                else:
                    method.on_task_switch(task_id)

            test_task_loader = self.test_dataloader(**self.dataloader_kwargs)

            # Manual test loop:
            from tqdm import tqdm
            pbar = tqdm(test_task_loader) 
            for batch_index, (x, y, task_labels) in enumerate(pbar):
                batch_size = len(x)

                if not self.task_labels_at_test_time:
                    # If the task labels aren't given at test time, then we
                    # give a list of Nones.
                    # TODO: This might cause problems when trying to move things
                    # between devices. We could instead not pass the argument,
                    # or set -1 in a tensor if the task label isn't given?
                    # If this works (passing a list of Nones), then this would
                    # also enable only giving a portion of the task labels,
                    # which could potentially be interesting.
                    task_labels = [None] * len(x)

                # TODO: This here might be an issue with keeping the parent
                # methods compatible! Need to think about this a bit.
                y_pred = method.predict(x=x, task_labels=task_labels)
                
                batch_metrics: Union[Metrics, float] = self.get_metrics(y_pred=y_pred, y=y)
                
                task_metrics += batch_metrics
                pbar.set_postfix(task_metrics.to_pbar_message())

            pbar.close()
            logger.info(f"Results on task {task_id}: {task_metrics}")
            metrics_for_each_task.append(task_metrics)

        for i, task_metrics in enumerate(metrics_for_each_task):
            logger.info(f"Test Results on task {i}: {task_metrics}")
            # TODO: Add back Wandb logging somehow, even though we're doing the
            # evaluation loop ourselves.
            if isinstance(task_metrics, Metrics):
                if method.trainer.logger:
                    method.trainer.logger.log_metrics(task_metrics.to_log_dict())            
        
        average_metrics: Metrics = sum(metrics_for_each_task) / len(metrics_for_each_task)
        logger.info(f"Average test metrics accross all the test tasks: {average_metrics}")
        return average_metrics

    def get_metrics(self,
                    y_pred: Tensor,
                    y: Tensor):
        """TODO: Calculate the "metric" used to evaluate the method, given the
        results of the forward pass and the contents of the batch.

        Args:
            forward_pass (Dict[str, Tensor]): The results of the forward pass. A
                dict containing Tensors.
            x (Tensor): The batch of samples that generated this forward pass.
            y (Tensor): The batch of labels associated with `x`.
            t (List[Optional[float]]): Optional task labels for each sample in
                `x`. None by default, as we don't assume to always have the task
                labels available. Is passed whenever task labels are available
                in the current context.
        """
        # Here we use the 'Metrics' object (`ClassificationMetrics` for
        # Classification, `RegressionMetrics` for continual regression (which is
        # supported but which we haven't tested that much yet.
        # We use these objects because they also give us the class accuracy and
        # confusion matrices for free, which we (will soon be) able use to
        # produce some nice wandb plots!
        from common.metrics import get_metrics
        return get_metrics(y_pred=y_pred, y=y)
        
        
    def evaluate_old(self, method: "Method") -> ClassIncrementalResults:
        """ NOTE: Refactoring this atm. See the 'newer' version above.
        
        Tests the method and returns the Results.

        Overwrite this to customize testing for your experimental setting.

        Returns:
            Results: A Results object for this particular setting.
        """
        from methods import Method
        from cl_trainer import CLTrainer
        
        method: Method
        trainer: CLTrainer = method.trainer

        assert isinstance(trainer, CLTrainer), (
            "WIP: Experimenting with defining the evaluation procedure in the CLTrainer. "
            "Please use a CLTrainer rather than a Trainer for now."
        )
        assert trainer.get_model() is method.model and method.model is not None
        assert trainer.datamodule is self
        task_losses: List[Loss] = trainer.test(
            # method.model,
            # datamodule=self,
            # verbose=True,
        )
        return ClassIncrementalResults(
            hparams=method.hparams,
            test_loss=sum(task_losses),
            task_losses=task_losses,
        )
        # Run the actual evaluation.
        test_dataloaders = self.test_dataloaders()
        # TODO: Here we are 'manually' evaluating on one test dataset at a time.
        # However, in pytorch_lightning, if a LightningModule's
        # `test_dataloaders` method returns more than a single dataloader, then
        # the Trainer takes care of evaluating on each dataset in sequence.
        # Here we basically do this manually, so that the trainer doesn't give
        # the `dataloader_idx` keyword argument to the eval_step() method on the
        # LightningModule.
        task_losses: List[Loss] = []
        for i, task_loader in enumerate(test_dataloaders):
            logger.info(f"Starting evaluation on task {i}.")
            self.current_task_id = i

            if self.task_labels_at_test_time:
                method.on_task_switch(i)

            test_outputs = method.trainer.test(
                test_dataloaders=task_loader,
                # BUG: we don't want to use the best model checkpoint because
                # that model might be s
                ckpt_path=None,
                verbose=False,
            )
            task_loss: Loss = self.extract_task_loss(test_outputs)
            task_metrics = task_loss.losses["classification"].metric
            assert task_metrics, task_loss
            logger.info(f"Results: {task_metrics}")
            task_losses.append(task_loss)

        model = method.model
        from methods.models import Model
        if isinstance(model, Model):
            hparams = model.hp
        else:
            hparams = model.hparams
        return self.results_class(
            hparams=hparams,
            test_loss=sum(task_losses),
            task_losses=task_losses,
        )
    

    def extract_task_loss(self, test_outputs: Union[Dict, List[Dict]]) -> Loss:
        # TODO: Refactor this, we need to have a clearer way of getting the
        # loss for each task.
        if isinstance(test_outputs, list):
            assert len(test_outputs) == 1
            test_outputs = test_outputs[0]
        
        if "loss_object" not in test_outputs:
            # TODO: Design a cleaner API for the evaluation setup.
            raise RuntimeError(
                "At the moment, a Method's test step needs to return "
                "a dict with a `Loss` object at key 'loss_object'. The "
                "metrics that the setting cares about are taken from that "
                "object in order to create the Results and determine the "
                "value of the Setting's 'objective'."
            )
        task_loss: Loss = test_outputs["loss_object"]
        return task_loss

    def make_train_cl_loader(self, dataset: _ContinuumDataset) -> _BaseCLLoader:
        """ Creates a train ClassIncremental object from continuum. """
        # TODO: Should we pass the common_transforms and test_transforms to cl loader?
        # TODO: This doesn't sound quite right.
        if self.train_transforms == self.val_transforms:
            common_transforms = self.train_transforms
            train_transforms = []
        else:
            logger.debug(f"Transforms: {self.transforms}")
            logger.debug(f"Train Transforms: {self.train_transforms}")
            logger.debug(f"Val Transforms: {self.val_transforms}")
            logger.debug(f"Test Transforms: {self.test_transforms}")
            common_transforms = self.train_transforms
            raise NotImplementedError("Don't know yet how to use the common_transforms and test_transforms here.")
     
        return ClassIncremental(
            dataset,
            nb_tasks=self.nb_tasks,
            increment=self.increment,
            initial_increment=self.initial_increment,
            class_order=self.class_order,
            common_transformations=common_transforms,
            train_transformations=train_transforms,
            
            train=True,  # a different loader for test
        )

    def make_test_cl_loader(self, dataset: _ContinuumDataset) -> _BaseCLLoader:
        """ Creates a test ClassIncremental object from continuum. """
        # TODO: This is a bit weird. Need to think about this (How we setup the
        # transforms when using Continuum datasets).
        if self.transforms == self.test_transforms:
            common_transforms = self.test_transforms
        else:
            raise NotImplementedError("Don't know yet how to use the common_transforms and test_transforms here.")
        
        return ClassIncremental(
            dataset,
            nb_tasks=self.nb_tasks,
            increment=self.test_increment,
            initial_increment=self.test_initial_increment,
            class_order=self.test_class_order,
            common_transformations=common_transforms,
            train=False,
        )

    @property
    def dataset_class(self) -> Type[_ContinuumDataset]:
        return self.available_datasets[self.dataset]

    def make_dataset(self,
                     data_dir: Path,
                     download: bool = True,
                     train: bool = True,
                     transform: Callable = None,
                     **kwargs) -> _ContinuumDataset:
        # TODO: #7 Use this method here to fix the errors that happen when trying
        # to create every single dataset from continuum. 
        return self.dataset_class(
            data_path=data_dir,
            download=download,
            **kwargs
        )

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
        
        self.cl_dataset = self.make_dataset(self.data_dir, download=False)
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
        env: DataLoader = PassiveEnvironment(dataset, **kwargs)
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
        env: DataLoader = PassiveEnvironment(dataset, **kwargs)
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
        env: DataLoader = PassiveEnvironment(dataset, **kwargs)
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

    def num_classes_in_task(self, task_id: int) -> Union[int, List[int]]:
        """ Returns the number of classes in the given task. """
        if isinstance(self.increment, list):
            return self.increment[task_id]
        return self.increment

    @property
    def num_classes_in_current_task(self) -> int:
        """ Returns the number of classes in the current task. """
        return self.num_classes_in_task(self._current_task_id)

    def current_task_classes(self, train: bool=True) -> List[int]:
        """ Gives back the labels present in the current task. """
        start_index = sum(
            self.num_classes_in_task(i) for i in range(self._current_task_id)
        )
        end_index = start_index + self.num_classes_in_task(self._current_task_id)
        if train:
            return self.class_order[start_index:end_index]
        else:
            return self.test_class_order[start_index:end_index]

    def extract_accuracy(self, test_outputs: Union[Dict, List[Dict]]) -> float:
        """ Extracts the supervised classification accuracy from the outputs of
        Trainer.test().

        TODO: This could perhaps be moved (along with some of the training & 
        evaluation routine) to a dedicated Trainer (e.g. the CLTrainer).
        """
        task_loss = self.extract_task_loss(test_outputs)
        task_metrics: ClassificationMetrics = self.extract_metric(task_loss)
        return task_metrics.accuracy

    def extract_metric(self, task_loss: Loss) -> Metrics:
        """Extracts a Metrics (ClassificationMetrics) object from the outputs of
        Trainer.test().
        """
        return task_loss.all_metrics()["classification"]