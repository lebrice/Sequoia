import itertools
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Type, TypeVar, Union

import gym
import numpy as np
import wandb
from continuum.datasets import (
    CIFAR10,
    CIFAR100,
    EMNIST,
    KMNIST,
    MNIST,
    QMNIST,
    CIFARFellowship,
    FashionMNIST,
    ImageNet100,
    ImageNet1000,
    MNISTFellowship,
    Synbols,
    _ContinuumDataset,
)
from continuum.scenarios import ClassIncremental, _BaseScenario
from continuum.tasks import TaskSet, concat, split_train_val
from gym import Space, spaces
from simple_parsing import choice, field, list_field
from torch import Tensor
from torch.utils.data import ConcatDataset, Dataset, Subset

from sequoia.common.config import Config
from sequoia.common.gym_wrappers import RenderEnvWrapper, TransformObservation, TransformReward
from sequoia.common.gym_wrappers.convert_tensors import add_tensor_support
from sequoia.common.gym_wrappers.convert_tensors import (
    add_tensor_support as tensor_space,
)
from sequoia.common.spaces import Image, Sparse, TypedDictSpace
from sequoia.common.transforms import Compose, Transforms
from sequoia.settings.assumptions.continual import ContinualAssumption
from sequoia.settings.base import Method, SettingABC
from sequoia.settings.sl import SLSetting
from sequoia.settings.sl.environment import PassiveEnvironment
from sequoia.settings.sl.wrappers import MeasureSLPerformanceWrapper
from sequoia.utils.generic_functions import move
from sequoia.utils.logging_utils import get_logger
from sequoia.utils.utils import flag

from .environment import (
    ContinualSLEnvironment,
    ContinualSLTestEnvironment,
    base_action_spaces,
    base_observation_spaces,
    base_reward_spaces,
)
from .objects import (
    Actions,
    ActionType,
    Observations,
    ObservationType,
    Rewards,
    RewardType,
)
from .results import ContinualSLResults
from .wrappers import relabel

logger = get_logger(__file__)

EnvironmentType = TypeVar("EnvironmentType", bound=ContinualSLEnvironment)


@dataclass
class ContinualSLSetting(SLSetting, ContinualAssumption):
    """ Continuous, Task-Agnostic, Continual Supervised Learning.
    
    This is *currently* the most "general" Supervised Continual Learning setting in
    Sequoia.

    - Data distribution changes smoothly over time.
    - Smooth transitions between "tasks"
    - No information about task boundaries or task identity (no task IDs)
    - Maximum of one 'epoch' through the environment.
    """

    # Class variables that hold the 'base' observation/action/reward spaces for the
    # available datasets.
    base_observation_spaces: ClassVar[Dict[str, gym.Space]] = base_observation_spaces
    base_action_spaces: ClassVar[Dict[str, gym.Space]] = base_action_spaces
    base_reward_spaces: ClassVar[Dict[str, gym.Space]] = base_reward_spaces

    # (NOTE: commenting out SLSetting.Observations as it is the same class
    # as Setting.Observations, and we want a consistent method resolution order.
    Observations: ClassVar[Type[Observations]] = Observations
    Actions: ClassVar[Type[Actions]] = Actions
    Rewards: ClassVar[Type[Rewards]] = Rewards

    Environment: ClassVar[Type[SLSetting.Environment]] = ContinualSLEnvironment[
        Observations, Actions, Rewards
    ]

    Results: ClassVar[Type[ContinualSLResults]] = ContinualSLResults

    # Class variable holding a dict of the names and types of all available
    # datasets.
    # TODO: Issue #43: Support other datasets than just classification
    available_datasets: ClassVar[Dict[str, Type[_ContinuumDataset]]] = {
        c.__name__.lower(): c
        for c in [
            CIFARFellowship,
            MNISTFellowship,
            ImageNet100,
            ImageNet1000,
            CIFAR10,
            CIFAR100,
            EMNIST,
            KMNIST,
            MNIST,
            QMNIST,
            FashionMNIST,
            Synbols,
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
    increment: Union[int, List[int]] = list_field(
        2, type=int, nargs="*", alias="n_classes_per_task"
    )
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

    # Wether task boundaries are smooth or not.
    smooth_task_boundaries: bool = flag(True)
    # Wether the context (task) variable is stationary or not.
    stationary_context: bool = flag(False)
    # Wether tasks share the same action space or not.
    # TODO: This will probably be moved into a different assumption.
    shared_action_space: Optional[bool] = None

    # TODO: Need to put num_workers in only one place.
    batch_size: int = field(default=32, cmd=False)
    num_workers: int = field(default=4, cmd=False)
    
    # When True, a Monitor-like wrapper will be applied to the training environment
    # and monitor the 'online' performance during training. Note that in SL, this will
    # also cause the Rewards (y) to be withheld until actions are passed to the `send`
    # method of the Environment.
    monitor_training_performance: bool = flag(False)

    def __post_init__(self):
        super().__post_init__()
        assert not self.has_setup_fit
        # Test values default to the same as train.
        self.test_increment = self.test_increment or self.increment
        self.test_initial_increment = (
            self.test_initial_increment or self.initial_increment
        )
        self.test_class_order = self.test_class_order or self.class_order

        # TODO: For now we assume a fixed, equal number of classes per task, for
        # sake of simplicity. We could take out this assumption, but it might
        # make things a bit more complicated.
        if isinstance(self.increment, list) and len(self.increment) == 1:
            self.increment = self.increment[0]
        if isinstance(self.test_increment, list) and len(self.test_increment) == 1:
            self.test_increment = self.test_increment[0]
        assert isinstance(self.increment, int)
        assert isinstance(self.test_increment, int)

        if isinstance(self.action_space, spaces.Discrete):
            base_action_space = self.base_action_spaces[self.dataset]
            n_classes = base_action_space.n
            self.class_order = self.class_order or list(range(n_classes))
            if self.nb_tasks:
                self.increment = n_classes // self.nb_tasks

        if not self.nb_tasks:
            base_action_space = self.base_action_spaces[self.dataset]
            if isinstance(base_action_space, spaces.Discrete):
                self.nb_tasks = base_action_space.n // self.increment

        assert self.nb_tasks != 0, self.nb_tasks

        # The 'scenarios' for train and test from continuum. (ClassIncremental for now).
        self.train_cl_loader: Optional[_BaseScenario] = None
        self.test_cl_loader: Optional[_BaseScenario] = None
        self.train_cl_dataset: Optional[_ContinuumDataset] = None
        self.test_cl_dataset: Optional[_ContinuumDataset] = None

        self.train_datasets: List[TaskSet] = []
        self.val_datasets: List[TaskSet] = []
        self.test_datasets: List[TaskSet] = []

        # This will be set by the Experiment, or passed to the `apply` method.
        # TODO: This could be a bit cleaner.
        self.config: Config
        # Default path to which the datasets will be downloaded.
        self.data_dir: Optional[Path] = None

        self.train_env: ContinualSLEnvironment = None  # type: ignore
        self.val_env: ContinualSLEnvironment = None  # type: ignore
        self.test_env: ContinualSLEnvironment = None  # type: ignore

        # BUG: These `has_setup_fit`, `has_setup_test`, `has_prepared_data` properties
        # aren't working correctly: they get set before the call to the function has
        # been executed, making it impossible to check those values from inside those
        # functions.
        self._has_prepared_data = False
        self._has_setup_fit = False
        self._has_setup_test = False

    def apply(
        self, method: Method["ContinualSLSetting"], config: Config = None
    ) -> ContinualSLResults:
        """Apply the given method on this setting to producing some results."""
        # TODO: It still isn't super clear what should be in charge of creating
        # the config, and how to create it, when it isn't passed explicitly.
        self.config = config or self._setup_config(method)
        assert self.config is not None

        method.configure(setting=self)

        # Run the main loop (defined in ContinualAssumption).
        # Basically does the following:
        # 1. Call method.fit(train_env, valid_env)
        # 2. Test the method on test_env.
        # Return the results, as reported by the test environment.
        results: ContinualSLResults = super().main_loop(method)
        logger.info(results.summary())
        method.receive_results(self, results=results)
        return results

    def train_dataloader(
        self, batch_size: int = 32, num_workers: Optional[int] = 4
    ) -> EnvironmentType:
        if not self.has_prepared_data:
            self.prepare_data()
        if not self.has_setup_fit:
            self.setup("fit")

        if self.train_env:
            self.train_env.close()

        batch_size = batch_size if batch_size is not None else self.batch_size
        num_workers = num_workers if num_workers is not None else self.num_workers

        dataset = self._make_train_dataset()
        # TODO: Add some kind of Wrapper around the dataset to make it
        # semi-supervised?
        env = self.Environment(
            dataset,
            hide_task_labels=(not self.task_labels_at_train_time),
            observation_space=self.observation_space,
            action_space=self.action_space,
            reward_space=self.reward_space,
            Observations=self.Observations,
            Actions=self.Actions,
            Rewards=self.Rewards,
            pin_memory=True,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            one_epoch_only=(not self.known_task_boundaries_at_train_time),
        )
        if self.config.device:
            # TODO: Put this before or after the image transforms?
            env = TransformObservation(env, f=partial(move, device=self.config.device))
            env = TransformReward(env, f=partial(move, device=self.config.device))

        if self.config.render:
            # Add a wrapper that calls 'env.render' at each step?
            env = RenderEnvWrapper(env)

        # NOTE: The transforms from `self.transforms` (the 'base' transforms) were
        # already added when creating the datasets and the CL scenario.
        train_specific_transforms = self.additional_transforms(self.train_transforms)
        if train_specific_transforms:
            env = TransformObservation(env, f=train_specific_transforms)

        if self.monitor_training_performance:
            env = MeasureSLPerformanceWrapper(
                env, first_epoch_only=True, wandb_prefix=f"Train/",
            )

        # NOTE: Quickfix for the 'dtype' of the TypedDictSpace perhaps getting lost
        # when transforms don't propagate the 'dtype' field.
        env.observation_space.dtype = self.Observations
        self.train_env = env
        return self.train_env

    def val_dataloader(
        self, batch_size: int = 32, num_workers: Optional[int] = 4
    ) -> EnvironmentType:
        if not self.has_prepared_data:
            self.prepare_data()
        if not self.has_setup_validate:
            self.setup("validate")

        if self.val_env:
            self.val_env.close()

        batch_size = batch_size if batch_size is not None else self.batch_size
        num_workers = num_workers if num_workers is not None else self.num_workers

        dataset = self._make_val_dataset()
        # TODO: Add some kind of Wrapper around the dataset to make it
        # semi-supervised?
        # TODO: Change the reward and action spaces to also use objects.
        env = self.Environment(
            dataset,
            hide_task_labels=(not self.task_labels_at_train_time),
            observation_space=self.observation_space,
            action_space=self.action_space,
            reward_space=self.reward_space,
            Observations=self.Observations,
            Actions=self.Actions,
            Rewards=self.Rewards,
            pin_memory=True,
            batch_size=batch_size,
            num_workers=num_workers,
            one_epoch_only=(not self.known_task_boundaries_at_train_time),
        )

        if self.config.device:
            # TODO: Put this before or after the image transforms?
            env = TransformObservation(env, f=partial(move, device=self.config.device))
            env = TransformReward(env, f=partial(move, device=self.config.device))
        # TODO: If wandb is enabled, then add customized Monitor wrapper (with
        # IterableWrapper as an additional subclass). There would then be a lot of
        # overlap between such a Monitor and the current TestEnvironment.
        if self.config.render:
            # Add a wrapper that calls 'env.render' at each step?
            env = RenderEnvWrapper(env)

        # NOTE: The transforms from `self.transforms` (the 'base' transforms) were
        # already added when creating the datasets and the CL scenario.
        val_specific_transforms = self.additional_transforms(self.val_transforms)
        if val_specific_transforms:
            env = TransformObservation(env, f=val_specific_transforms)

        # NOTE: We don't measure online performance on the validation set.
        # if self.monitor_training_performance:
        #     env = MeasureSLPerformanceWrapper(
        #         env,
        #         first_epoch_only=True,
        #         wandb_prefix=f"Train/Task {self.current_task_id}",
        #     )

        # NOTE: Quickfix for the 'dtype' of the TypedDictSpace perhaps getting lost
        # when transforms don't propagate the 'dtype' field.
        env.observation_space.dtype = self.Observations
        self.val_env = env
        return self.val_env

    def test_dataloader(
        self, batch_size: int = None, num_workers: int = None
    ) -> ContinualSLEnvironment[Observations, Actions, Rewards]:
        """ Returns a Continual SL Test environment.
        """
        if not self.has_prepared_data:
            self.prepare_data()
        if not self.has_setup_test:
            self.setup("test")

        batch_size = batch_size if batch_size is not None else self.batch_size
        num_workers = num_workers if num_workers is not None else self.num_workers

        dataset = self._make_test_dataset()
        env = self.Environment(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            hide_task_labels=(not self.task_labels_at_test_time),
            observation_space=self.observation_space,
            action_space=self.action_space,
            reward_space=self.reward_space,
            Observations=self.Observations,
            Actions=self.Actions,
            Rewards=self.Rewards,
            pretend_to_be_active=True,
            shuffle=False,
            one_epoch_only=True,
        )
        
        if self.config.device:
            # TODO: Put this before or after the image transforms?
            env = TransformObservation(env, f=partial(move, device=self.config.device))
            env = TransformReward(env, f=partial(move, device=self.config.device))

        # NOTE: The transforms from `self.transforms` (the 'base' transforms) were
        # already added when creating the datasets and the CL scenario.
        test_specific_transforms = self.additional_transforms(self.test_transforms)
        if test_specific_transforms:
            env = TransformObservation(env, f=test_specific_transforms)

        # FIXME: Instead of trying to create a 'fake' task schedule for the test
        # environment, instead let the test environment see the task ids, (and then hide
        # them if necessary) so that it can compile the stats for each task based on the
        # task IDs of the observations.

        # TODO: Configure the 'monitoring' dir properly.
        if wandb.run:
            test_dir = wandb.run.dir
        else:
            test_dir = "results"

        test_loop_max_steps = len(dataset) // (env.batch_size or 1)
        test_env = ContinualSLTestEnvironment(
            env,
            directory=test_dir,
            step_limit=test_loop_max_steps,
            force=True,
            config=self.config,
            video_callable=None if (wandb.run or self.config.render) else False,
        )

        # NOTE: Quickfix for the 'dtype' of the TypedDictSpace perhaps getting lost
        # when transforms don't propagate the 'dtype' field.
        env.observation_space.dtype = self.Observations
        if self.test_env:
            self.test_env.close()
        self.test_env = test_env
        return self.test_env

    def prepare_data(self, data_dir: Path = None) -> None:
        # TODO: Pass the transformations to the CL scenario, or to the dataset?
        if data_dir is None:
            if self.config:
                data_dir = self.config.data_dir
            else:
                data_dir = Path("data")

        logger.info(f"Downloading datasets to directory {data_dir}")
        self.train_cl_dataset = self.make_dataset(data_dir, download=True, train=True)
        self.test_cl_dataset = self.make_dataset(data_dir, download=True, train=False)
        return super().prepare_data()

    def setup(self, stage: str = None):
        if not self.has_prepared_data:
            self.prepare_data()
        super().setup(stage=stage)

        if stage not in (None, "fit", "test", "validate"):
            raise RuntimeError(f"`stage` should be 'fit', 'test', 'validate' or None.")

        if stage in (None, "fit", "validate"):
            self.train_cl_dataset = self.train_cl_dataset or self.make_dataset(
                self.config.data_dir, download=False, train=True
            )
            self.train_cl_loader = self.train_cl_loader or ClassIncremental(
                cl_dataset=self.train_cl_dataset,
                nb_tasks=self.nb_tasks,
                increment=self.increment,
                initial_increment=self.initial_increment,
                transformations=self.train_transforms,
                class_order=self.class_order,
            )
            if not self.train_datasets and not self.val_datasets:
                for task_id, train_taskset in enumerate(self.train_cl_loader):
                    train_taskset, valid_taskset = split_train_val(
                        train_taskset, val_split=0.1
                    )
                    self.train_datasets.append(train_taskset)
                    self.val_datasets.append(valid_taskset)
                # IDEA: We could do the remapping here instead of adding a wrapper later.
                if self.shared_action_space and isinstance(
                    self.action_space, spaces.Discrete
                ):
                    # If we have a shared output space, then they are all mapped to [0, n_per_task]
                    self.train_datasets = list(map(relabel, self.train_datasets))
                    self.val_datasets = list(map(relabel, self.val_datasets))

        if stage in (None, "test"):
            self.test_cl_dataset = self.test_cl_dataset or self.make_dataset(
                self.config.data_dir, download=False, train=False
            )
            self.test_cl_loader = self.test_cl_loader or ClassIncremental(
                cl_dataset=self.test_cl_dataset,
                nb_tasks=self.nb_tasks,
                increment=self.test_increment,
                initial_increment=self.test_initial_increment,
                transformations=self.test_transforms,
                class_order=self.test_class_order,
            )
            if not self.test_datasets:
                # TODO: If we decide to 'shuffle' the test tasks, then store the sequence of
                # task ids in a new property, probably here.
                # self.test_task_order = list(range(len(self.test_datasets)))
                self.test_datasets = list(self.test_cl_loader)
                # IDEA: We could do the remapping here instead of adding a wrapper later.
                if self.shared_action_space and isinstance(
                    self.action_space, spaces.Discrete
                ):
                    # If we have a shared output space, then they are all mapped to [0, n_per_task]
                    self.test_datasets = list(map(relabel, self.test_datasets))

    def _make_train_dataset(self) -> Union[TaskSet, Dataset]:
        # NOTE: Passing the same seed to `train`/`valid`/`test` is fine, because it's
        # only used for the shuffling used to make the task boundaries smooth.
        if self.smooth_task_boundaries:
            return smooth_task_boundaries_concat(
                self.train_datasets, seed=self.config.seed if self.config else None
            )
        if self.stationary_context:
            joined_dataset = concat(self.train_datasets)
            return shuffle(joined_dataset, seed=self.config.seed)
        if self.known_task_boundaries_at_train_time:
            return self.train_datasets[self.current_task_id]
        else:
            return concat(self.train_datasets)

    def _make_val_dataset(self) -> Dataset:
        if self.smooth_task_boundaries:
            return smooth_task_boundaries_concat(
                self.val_datasets, seed=self.config.seed
            )
        if self.stationary_context:
            joined_dataset = concat(self.val_datasets)
            return shuffle(joined_dataset, seed=self.config.seed)
        if self.known_task_boundaries_at_train_time:
            return self.val_datasets[self.current_task_id]
        return concat(self.val_datasets)

    def _make_test_dataset(self) -> Dataset:
        if self.smooth_task_boundaries:
            return smooth_task_boundaries_concat(
                self.test_datasets, seed=self.config.seed
            )
        else:
            return concat(self.test_datasets)

    def make_dataset(
        self, data_dir: Path, download: bool = True, train: bool = True, **kwargs
    ) -> _ContinuumDataset:
        # TODO: #7 Use this method here to fix the errors that happen when
        # trying to create every single dataset from continuum.
        data_dir = Path(data_dir)

        if not data_dir.exists():
            data_dir.mkdir(parents=True, exist_ok=True)

        if self.dataset in self.available_datasets:
            dataset_class = self.available_datasets[self.dataset]
            return dataset_class(
                data_path=data_dir, download=download, train=train, **kwargs
            )

        elif self.dataset in self.available_datasets.values():
            dataset_class = self.dataset
            return dataset_class(
                data_path=data_dir, download=download, train=train, **kwargs
            )

        elif isinstance(self.dataset, Dataset):
            logger.info(f"Using a custom dataset {self.dataset}")
            return self.dataset

        else:
            raise NotImplementedError

    @property
    def observation_space(self) -> TypedDictSpace[Observations]:
        """ The un-batched observation space, based on the choice of dataset and
        the transforms at `self.transforms` (which apply to the train/valid/test
        environments).

        The returned space is a TypedDictSpace, with the following properties:
        - `x`: observation space (e.g. `Image` space)
        - `task_labels`: Union[Discrete, Sparse[Discrete]]
           The task labels for each sample. When task labels are not available,
           the task labels space is Sparse, and entries will be `None`.
           
        TODO: Replace this property's type with a `Space[Observations]` (and also create
        this `Space` generic)
        """
        x_space = self.base_observation_spaces[self.dataset]
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

        return TypedDictSpace(
            x=x_space, task_labels=task_label_space, dtype=self.Observations,
        )

    # TODO: Add a `train_observation_space`, `train_action_space`, `train_reward_space`?

    @property
    def action_space(self) -> spaces.Discrete:
        """ Action space for this setting. """
        base_action_space = self.base_action_spaces[self.dataset]
        if isinstance(base_action_space, spaces.Discrete):
            if self.shared_action_space:
                assert isinstance(self.increment, int), (
                    "Need to have same number of classes in each task when "
                    "`shared_action_space` is true."
                )
                return spaces.Discrete(self.increment)
        return base_action_space

        # TODO: IDEA: Have the action space only reflect the number of 'current' classes
        # in order to create a "true" class-incremental learning setting.
        # n_classes_seen_so_far = 0
        # for task_id in range(self.current_task_id):
        #     n_classes_seen_so_far += self.num_classes_in_task(task_id)
        # return spaces.Discrete(n_classes_seen_so_far)

    @property
    def reward_space(self) -> spaces.Discrete:
        base_reward_space = self.base_action_spaces[self.dataset]
        if isinstance(base_reward_space, spaces.Discrete):
            if self.shared_action_space:
                assert isinstance(self.increment, int), (
                    "Need to have same number of classes in each task when "
                    "`shared_action_space` is true."
                )
                return spaces.Discrete(self.increment)
        return base_reward_space

    def additional_transforms(self, stage_transforms: List[Transforms]) -> Compose:
        """ Returns the transforms in `stage_transforms` that are additional transforms
        from those in `self.transforms`.

        For example, if:
        ```
        setting.transforms = Compose([Transforms.Resize(32), Transforms.ToTensor])
        setting.train_transforms = Compose([Transforms.Resize(32), Transforms.ToTensor, Transforms.RandomGrayscale])
        ```
        Then:
        ```
        setting.additional_transforms(setting.train_transforms)
        # will give:
        Compose([Transforms.RandomGrayscale])
        ```
        """
        reference_transforms = self.transforms

        if len(stage_transforms) < len(reference_transforms):
            # Assume no overlap, return all the 'stage' transforms.
            return Compose(stage_transforms)
        if stage_transforms == reference_transforms:
            # Complete overlap, return an empty list.
            return Compose([])

        # IDEA: Only add the additional transforms, compared to the 'base' transforms.
        # As soon as one is different, break.
        i = 0
        for i, t_a, t_b in enumerate(zip(stage_transforms, self.transforms)):
            if t_a != t_b:
                break
        return Compose(stage_transforms[i:])


def smooth_task_boundaries_concat(
    datasets: List[Dataset], seed: int = None, window_length: float = 0.03
) -> ConcatDataset:
    """ TODO: Use a smarter way of mixing from one to the other? """
    lengths = [len(dataset) for dataset in datasets]
    total_length = sum(lengths)
    n_tasks = len(datasets)

    if not isinstance(window_length, int):
        window_length = int(total_length * window_length)
    assert (
        window_length > 1
    ), f"Window length should be positive or a fraction of the dataset length. ({window_length})"

    rng = np.random.default_rng(seed)

    def option1():
        shuffled_indices = np.arange(total_length)
        for start_index in range(
            0, total_length - window_length + 1, window_length // 2
        ):
            rng.shuffle(shuffled_indices[start_index : start_index + window_length])
        return shuffled_indices

    # Maybe do the same but backwards?

    # IDEA #2: Sample based on how close to the 'center' of the task we are.
    def option2():
        boundaries = np.array(list(itertools.accumulate(lengths, initial=0)))
        middles = [
            (start + end) / 2 for start, end in zip(boundaries[0:], boundaries[1:])
        ]
        samples_left: Dict[int, int] = {i: length for i, length in enumerate(lengths)}
        indices_left: Dict[int, List[int]] = {
            i: list(range(boundaries[i], boundaries[i] + length))
            for i, length in enumerate(lengths)
        }

        out_indices: List[int] = []
        last_dataset_index = n_tasks - 1
        for step in range(total_length):
            if step < middles[0] and samples_left[0]:
                # Prevent sampling things from task 1 at the beginning of task 0, and
                eligible_dataset_ids = [0]
            elif step > middles[-1] and samples_left[last_dataset_index]:
                # Prevent sampling things from task N-1 at the emd of task N
                eligible_dataset_ids = [last_dataset_index]
            else:
                # 'smooth', but at the boundaries there are actually two or three datasets,
                # from future tasks even!
                eligible_dataset_ids = list(k for k, v in samples_left.items() if v > 0)
                # if len(eligible_dataset_ids) > 2:
                #     # Prevent sampling from future tasks (past the next task) when at a
                #     # boundary.
                #     left_dataset_index = min(eligible_dataset_ids)
                #     right_dataset_index = min(
                #         v for v in eligible_dataset_ids if v > left_dataset_index
                #     )
                #     eligible_dataset_ids = [left_dataset_index, right_dataset_index]

            options = np.array(eligible_dataset_ids, dtype=int)

            # Calculate the 'distance' to the center of the task's dataset.
            distances = np.abs(
                [step - middles[dataset_index] for dataset_index in options]
            )

            # NOTE: THis exponent is kindof arbitrary, setting it to this value because it
            # sortof works for MNIST so far.
            probs = 1 / (1 + np.abs(distances) ** 2)
            probs /= sum(probs)

            chosen_dataset = rng.choice(options, p=probs)
            chosen_index = indices_left[chosen_dataset].pop()
            samples_left[chosen_dataset] -= 1
            out_indices.append(chosen_index)

        shuffled_indices = np.array(out_indices)
        return shuffled_indices

    def option3():
        shuffled_indices = np.arange(total_length)
        for start_index in range(
            0, total_length - window_length + 1, window_length // 2
        ):
            rng.shuffle(shuffled_indices[start_index : start_index + window_length])
        for start_index in reversed(
            range(0, total_length - window_length + 1, window_length // 2)
        ):
            rng.shuffle(shuffled_indices[start_index : start_index + window_length])
        return shuffled_indices

    shuffled_indices = option3()

    if all(isinstance(dataset, TaskSet) for dataset in datasets):
        # Use the 'concat' from continuum, just to preserve the field/methods of a
        # TaskSet.
        joined_taskset = concat(datasets)
        return subset(joined_taskset, shuffled_indices)
    else:
        joined_dataset = ConcatDataset(datasets)
        return Subset(joined_dataset, shuffled_indices)

    return shuffled_indices


from functools import singledispatch
from typing import Sequence, overload

from .wrappers import replace_taskset_attributes

DatasetType = TypeVar("DatasetType", bound=Dataset)


@overload
def subset(dataset: TaskSet, indices: Sequence[int]) -> TaskSet:
    ...


@singledispatch
def subset(dataset: DatasetType, indices: Sequence[int]) -> Union[Subset, DatasetType]:
    raise NotImplementedError(f"Don't know how to take a subset of dataset {dataset}")
    return Subset(dataset, indices)


@subset.register
def taskset_subset(taskset: TaskSet, indices: np.ndarray) -> TaskSet:
    # x, y, t = taskset.get_raw_samples(indices)
    x, y, t = taskset.get_raw_samples(indices)
    # TODO: Not sure if/how to handle the `bounding_boxes` attribute here.
    bounding_boxes = taskset.bounding_boxes
    if bounding_boxes is not None:
        bounding_boxes = bounding_boxes[indices]
    return replace_taskset_attributes(
        taskset, x=x, y=y, t=t, bounding_boxes=bounding_boxes
    )


def random_subset(
    taskset: TaskSet, n_samples: int, seed: int = None, ordered: bool = True
) -> TaskSet:
    """ Returns a random (ordered) subset of the given TaskSet. """
    rng = np.random.default_rng(seed)
    dataset_length = len(taskset)
    if n_samples > dataset_length:
        raise RuntimeError(
            f"Dataset has {dataset_length}, asked for {n_samples} samples."
        )
    indices = rng.permutation(range(dataset_length))[:n_samples]
    # indices = rng.choice(len(taskset), size=n_samples, replace=False)
    if ordered:
        indices = sorted(indices)
    assert len(indices) == n_samples
    return subset(taskset, indices)


DatasetType = TypeVar("DatasetType", bound=Dataset)


def shuffle(dataset: DatasetType, seed: int = None) -> DatasetType:
    length = len(dataset)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(range(length))
    return subset(dataset, perm)
