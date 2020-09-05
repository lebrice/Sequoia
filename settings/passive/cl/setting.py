import warnings
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import (Any, Callable, ClassVar, Dict, List, Optional, Tuple, Type,
                    TypeVar, Union)

from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader

from common import ClassificationMetrics, Metrics
from common.config import Config
from common.dims import Dims
from common.loss import Loss
from common.metrics import ClassificationMetrics
from common.transforms import Compose, Transforms
from continuum import ClassIncremental, split_train_val
from continuum.datasets import *
from continuum.datasets import _ContinuumDataset
from continuum.scenarios.base import _BaseCLLoader
from settings.base import Results
from settings.base.environment import ObservationType, RewardType
from settings.passive.environment import PassiveEnvironment
from simple_parsing import (Serializable, choice, field, list_field,
                            mutable_field)
from utils import dict_union, get_logger
from utils.utils import constant

from ..environment import PassiveEnvironment
from ..setting import PassiveSetting
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
    """Settings where the data is online non-stationary.

    At the moment, this is basically just a base class for the Task-Incremental
    Setting (TaskIncrementalSetting) where the data is split up into 'tasks'.

    However in the future, as we add more CL setups, they should extend this
    class, and we might need to move some of the stuff here into
    TaskIncremental if needed.

    For example, we might want to create something like a 'stream' learning of
    some sort, where the transitions are smooth and there are no task labels.

    This implements the LightningDataModule API from pytorch-lightning-bolts.
    The hope is that this greatly simplifies the whole data generation process.
    - `train_dataloader`, `val_dataloader` and `test_dataloader` give
        dataloaders of the current task.
    - `train_dataloaders`, `val_dataloaders` and `test_dataloaders` give the 
        dataloaders of all the tasks. NOTE: this isn't part of the
        LightningDataModule API.

    The current task can be set at the `current_task_id` attribute.

    TODO: Change the base class from PassiveSetting to `ActiveSetting` for
    continual active learning / continual RL.
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

    # Wether the current task id can be read from outside this class.
    # NOTE: Loosely enforced, could be bypassed if people want to 'cheat'.
    # TODO: Adding a mechanism for making task label only available at train time?
    task_label_is_readable: bool = True
    # Wether the current task id can be set from outside this class.
    task_label_is_writable: bool = True

    # Transformations to use. See the Transforms enum for the available values.
    transforms: List[Transforms] = list_field(Transforms.to_tensor, Transforms.fix_channels, to_dict=False)
    
    # Wether task labels are available at train time. (Forced to True.)
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
        self._dims: Tuple[int, int, int] = dims_for_dataset[self.dataset]
        self._num_classes: int = num_classes_in_dataset[self.dataset]

        super().__post_init__(
            obs_shape=self._dims,
            action_shape=self._num_classes,
            reward_shape=self._num_classes,
        )

        self._current_task_id: int = 0

        self.train_dataset: _ContinuumDataset = None
        self.test_dataset: _ContinuumDataset = None
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
        method: Method
        trainer = method.trainer

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
    
    def extract_accuracy(self, test_outputs: Union[Dict, List[Dict]]) -> float:
        task_loss = self.extract_task_loss(test_outputs)
        task_metrics: ClassificationMetrics = self.extract_metric(task_loss)
        return task_metrics.accuracy

    def extract_metric(self, task_loss: Loss) -> Metrics:
        return task_loss.all_metrics()["classification"]

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
            train=True,  # a different loader for test
        )

    def make_test_cl_loader(self, dataset: _ContinuumDataset) -> _BaseCLLoader:
        """ Creates a test ClassIncremental object from continuum. """
        return ClassIncremental(
            dataset,
            nb_tasks=self.nb_tasks,
            increment=self.test_increment,
            initial_increment=self.test_initial_increment,
            class_order=self.test_class_order,
            # TODO: Figure out what/how to pass in test transforms, if it
            # makes any sense to do so.
            common_transformations=self.transforms,
            train=False  # a different loader for test
        )

    @property
    def dataset_class(self) -> Type[_ContinuumDataset]:
        return type(self).available_datasets[self.dataset]

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

    @property
    def dims(self) -> Dims:
        """Gets the dimensions of the input, taking into account the transforms.
        
        TODO: Could transforms just specify their impact on the shape directly
        instead, à-la Tensorflow? (with some kind of class method)?
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

    def train_dataloaders(self, **kwargs) -> List[PassiveEnvironment]:
        """Returns the DataLoaders for all the train datasets. """
        loaders: List[DataLoader] = []
        for i, dataset in enumerate(self.train_datasets):
            kwargs = dict_union(self.dataloader_kwargs, kwargs)
            env: DataLoader = PassiveEnvironment(dataset, **kwargs)
            loaders.append(env)
        return loaders

    def test_dataloaders(self, **kwargs) -> List[PassiveEnvironment]:
        """Returns the DataLoaders for all the test datasets. """
        loaders: List[DataLoader] = []
        for i, dataset in enumerate(self.test_datasets):
            kwargs = dict_union(self.dataloader_kwargs, kwargs)
            env: DataLoader = PassiveEnvironment(dataset, **kwargs)
            loaders.append(env)
        return loaders

    def val_dataloaders(self, **kwargs) -> List[PassiveEnvironment]:
        """Returns the DataLoaders for all the validation datasets. """
        loaders: List[DataLoader] = []
        for i, dataset in enumerate(self.val_datasets):
            kwargs = dict_union(self.dataloader_kwargs, kwargs)
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

    def num_classes_in_task(self, task_id: int) -> Union[int, List[int]]:
        if isinstance(self.increment, list):
            return self.increment[task_id]
        return self.increment

    @property
    def num_classes_in_current_task(self) -> int:
        return self.num_classes_in_task(self._current_task_id)

    def current_task_classes(self, train: bool=True) -> List[int]:
        start_index = sum(
            self.num_classes_in_task(i) for i in range(self._current_task_id)
        )
        end_index = start_index + self.num_classes_in_task(self._current_task_id)
        if train:
            return self.class_order[start_index:end_index]
        else:
            return self.test_class_order[start_index:end_index]
