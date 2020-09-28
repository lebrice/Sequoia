""" Define a "Class-Incremental" Continual Learning Setting.

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

The Setting class is based on the `LightningDataModule` from
pl_bolts (pytorch-lightning-bolts). The hope is that by staying close to that
API, we can reuse some of the models that people develop while target that API.
- `train_dataloader`, `val_dataloader` and `test_dataloader` give
    dataloaders of the current task.
- `train_dataloaders`, `val_dataloaders` and `test_dataloaders` give the 
    dataloaders of all the tasks. NOTE: this isn't part of the
    LightningDataModule API.

"""
import warnings
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import (Callable, ClassVar, Dict, List, Optional, Tuple, Type, Union)

from simple_parsing import choice, list_field
from torch import Tensor
from torch.utils.data import DataLoader

from common import ClassificationMetrics, Metrics
from common.config import Config
from common.loss import Loss
from common.transforms import Transforms
from continuum import ClassIncremental, split_train_val
from continuum.datasets import *
from continuum.datasets import _ContinuumDataset
from continuum.scenarios.base import _BaseCLLoader
from settings.base import Results
from settings.base.environment import ObservationType, RewardType
from settings.passive.environment import PassiveEnvironment
from utils import dict_union, get_logger
from utils.utils import constant

from ..environment import PassiveEnvironment
from ..passive_setting import PassiveSetting
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

    # Transformations to use. See the Transforms enum for the available values.
    transforms: List[Transforms] = list_field(Transforms.to_tensor, Transforms.fix_channels, to_dict=False)
    
    # Wether task labels are available at train time.
    # NOTE: Forced to True at the moment.
    task_labels_at_train_time: bool = constant(True)
    # Wether task labels are available at test time.
    task_labels_at_test_time: bool = False

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
        """Creates a new CL environment / setup.

        Args:
            options (Options): Dataclass used for configuration.
        """
        assert self.dataset in dims_for_dataset, f"{self.dataset}, {dims_for_dataset.keys()}"
        if not hasattr(self, "num_classes"):
            # In some concrete LightningDataModule's like MnistDataModule,
            # num_classes is a read-only property. Therefore we check if it
            # is already defined. This is just in case something tries to
            # inherit from both IIDSetting and MnistDataModule, for instance.
            self.num_classes: int = num_classes_in_dataset[self.dataset]
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
        self._setup = False
        self._configured = False
        self._prepared = False
        self.data_dir: Optional[Path] = None

        if isinstance(self.increment, list) and len(self.increment) == 1:
            self.increment = self.increment[0]
        if self.nb_tasks == 0:
            self.nb_tasks = num_classes_in_dataset[self.dataset] // self.increment
        else:
            self.increment = num_classes_in_dataset[self.dataset] // self.nb_tasks
        if not self.class_order:
            self.class_order = list(range(self.num_classes))
        # Test values default to the same as train.
        self.test_increment = self.test_increment or self.increment
        self.test_initial_increment = self.test_initial_increment or self.test_increment
        self.test_class_order = self.test_class_order or self.class_order

    def evaluate(self, method: "Method") -> ClassIncrementalResults:
        """Tests the method and returns the Results.

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

    def prepare_data(self, data_dir: Path=None, **kwargs):
        data_dir = data_dir or self.data_dir
        assert data_dir, "One of self.data_dir or the data_dir keyword argument to setup() must be set!"
        if self.data_dir is None:
            self.data_dir = data_dir
        self.make_dataset(data_dir, download=True)
        super().prepare_data(**kwargs)
        self._prepared = True

    def setup(self, *args, **kwargs):
        """ Creates the datasets for each task.
        
        TODO: Figure out a way of setting data_dir elsewhere maybe?
        """
        if not self._configured:
            self.configure(config=self.config or Config())
            self._configured = True
        
        # TODO: Data should be prepared..
        if self.config.device.type == "cpu" and not self._prepared:
            self.prepare_data()
            self._prepared = True

        assert self.data_dir, "One of self.data_dir or the data_dir keyword argument to setup() must be set!"
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

        super().setup(*args, **kwargs)
        self._setup = True

    def train_dataloader(self, **kwargs) -> PassiveEnvironment:
        """Returns a DataLoader for the train dataset of the current task.
        
        NOTE: The dataloader is passive for now (just a regular DataLoader).
        """
        # self.setup_if_needed()
        dataset = self.train_datasets[self._current_task_id]
        kwargs = dict_union(self.dataloader_kwargs, kwargs)
        env: DataLoader = PassiveEnvironment(dataset, **kwargs)
        return env

    def val_dataloader(self, **kwargs) -> PassiveEnvironment:
        """Returns a DataLoader for the validation dataset of the current task.
        
        NOTE: The dataloader is passive for now (just a regular DataLoader).
        """
        self.setup_if_needed()
        dataset = self.val_datasets[self._current_task_id]
        kwargs = dict_union(self.dataloader_kwargs, kwargs)
        env: DataLoader = PassiveEnvironment(dataset, **kwargs)
        return env

    def test_dataloader(self, **kwargs) -> PassiveEnvironment:
        """Returns a DataLoader for the test dataset of the current task.
        
        NOTE: The dataloader is passive for now (just a regular DataLoader).
        """
        self.setup_if_needed()
        dataset = self.test_datasets[self._current_task_id]
        kwargs = dict_union(self.dataloader_kwargs, kwargs)
        env: DataLoader = PassiveEnvironment(dataset, **kwargs)
        return env

    """
    NOTE: Currently trying to stick closer to the LightningDataModule API, and
    instead of giving back the loaders for all the tasks at once in
    `val_dataloader` and in `test_dataloader` (which would give the task ids
    to the model indirectly with the dataloader_idx argument to the `val_step`
    and `test_step` methods)
    do something more like:
    
    ```
    # Training loop:
    for task_id in range(nb_tasks):
        setting.current_task_id = task_id

        if setting.task_labels_at_train_time:
            model.on_task_switch(task_id)
        
        success = trainer.fit(model, datamodule=setting)
        
        # Test loop:
        for test_task_id in range(nb_tasks):
            setting.current_task_id = test_task_id
            if setting.task_labels_at_test_time:
                model.on_task_switch(test_task_id)

            results_after_learning_task_i = trainer.test(model, test_dataloaders=setting.test_dataloaders())
    ```
    """


    # def train_dataloaders(self, **kwargs) -> List[PassiveEnvironment]:
    #     """Returns the DataLoaders for all the train datasets. """
    #     loaders: List[DataLoader] = []
    #     for i, dataset in enumerate(self.train_datasets):
    #         kwargs = dict_union(self.dataloader_kwargs, kwargs)
    #         env: DataLoader = PassiveEnvironment(dataset, **kwargs)
    #         loaders.append(env)
    #     return loaders

    # def test_dataloaders(self, **kwargs) -> List[PassiveEnvironment]:
    #     """Returns the DataLoaders for all the test datasets. """
    #     loaders: List[DataLoader] = []
    #     for i, dataset in enumerate(self.test_datasets):
    #         kwargs = dict_union(self.dataloader_kwargs, kwargs)
    #         env: DataLoader = PassiveEnvironment(dataset, **kwargs)
    #         loaders.append(env)
    #     return loaders

    # def val_dataloaders(self, **kwargs) -> List[PassiveEnvironment]:
    #     """Returns the DataLoaders for all the validation datasets. """
    #     loaders: List[DataLoader] = []
    #     for i, dataset in enumerate(self.val_datasets):
    #         kwargs = dict_union(self.dataloader_kwargs, kwargs)
    #         env: DataLoader = PassiveEnvironment(dataset, **kwargs)
    #         loaders.append(env)
    #     return loaders

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