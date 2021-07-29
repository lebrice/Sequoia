""" Defines a `Setting` subclass for "Class-Incremental" Continual Learning.

Example command to run a method on this setting (in debug mode):
```
python main.py --setting class_incremental --method baseline --debug  \
    --batch_size 128 --max_epochs 1
```

Class-Incremental definition from [iCaRL](https://arxiv.org/abs/1611.07725):

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Type, Union
import wandb
import gym
import numpy as np
import torch
from continuum import ClassIncremental
from continuum.datasets import (
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
)
from continuum.datasets import _ContinuumDataset
from continuum.scenarios.base import _BaseScenario
from continuum.tasks import split_train_val
from gym import Space, spaces
from simple_parsing import choice, field, list_field
from torch import Tensor
from torch.utils.data import ConcatDataset, Dataset

from sequoia.common.config import Config
from sequoia.common.gym_wrappers import TransformObservation
from sequoia.settings.assumptions.incremental import (
    IncrementalAssumption,
    IncrementalResults,
)
from sequoia.settings.base import Method, Results
from sequoia.utils import get_logger

from sequoia.settings.sl.environment import Actions, PassiveEnvironment, Rewards
from sequoia.settings.sl.setting import SLSetting
from sequoia.settings.sl.continual import ContinualSLSetting
from sequoia.settings.sl.continual.wrappers import relabel
from sequoia.settings.sl.wrappers import MeasureSLPerformanceWrapper
from sequoia.settings.rl.wrappers import HideTaskLabelsWrapper
from continuum.tasks import concat

from ..continual import ContinualSLTestEnvironment
from ..discrete.setting import DiscreteTaskAgnosticSLSetting
from .results import IncrementalSLResults
from .environment import IncrementalSLEnvironment, IncrementalSLTestEnvironment
from .objects import (
    Observations,
    ObservationType,
    Actions,
    ActionType,
    Rewards,
    RewardType,
)

logger = get_logger(__file__)
# # NOTE: This dict reflects the observation space of the different datasets
# # *BEFORE* any transforms are applied. The resulting property on the Setting is
# # based on this 'base' observation space, passed through the transforms.
# # TODO: Make it possible to automatically add tensor support if the dtype passed to a
# # gym space is a `torch.dtype`.
# tensor_space = add_tensor_support


@dataclass
class IncrementalSLSetting(IncrementalAssumption, DiscreteTaskAgnosticSLSetting):
    """Supervised Setting where the data is a sequence of 'tasks'.

    This class is basically is the supervised version of an Incremental Setting


    The current task can be set at the `current_task_id` attribute.
    """

    Results: ClassVar[Type[IncrementalResults]] = IncrementalSLResults

    Observations: ClassVar[Type[Observations]] = Observations
    Actions: ClassVar[Type[Actions]] = Actions
    Rewards: ClassVar[Type[Rewards]] = Rewards

    Environment: ClassVar[Type[SLSetting.Environment]] = IncrementalSLEnvironment[
        Observations, Actions, Rewards
    ]

    Results: ClassVar[Type[IncrementalSLResults]] = IncrementalSLResults

    # Class variable holding a dict of the names and types of all available
    # datasets.
    available_datasets: ClassVar[Dict[str, Type[_ContinuumDataset]]] = DiscreteTaskAgnosticSLSetting.available_datasets.copy()

    # A continual dataset to use. (Should be taken from the continuum package).
    dataset: str = choice(available_datasets.keys(), default="mnist")

    # TODO: IDEA: Adding these fields/constructor arguments so that people can pass a
    # custom ready-made `Scenario` from continuum to use (not sure this is a good idea
    # though)
    train_cl_scenario: Optional[_BaseScenario] = field(
        default=None, cmd=False, to_dict=False
    )
    test_cl_scenario: Optional[_BaseScenario] = field(
        default=None, cmd=False, to_dict=False
    )

    def __post_init__(self):
        """Initializes the fields of the Setting (and LightningDataModule),
        including the transforms, shapes, etc.
        """
        super().__post_init__()

        # TODO: For now we assume a fixed, equal number of classes per task, for
        # sake of simplicity. We could take out this assumption, but it might
        # make things a bit more complicated.
        assert isinstance(self.increment, int)
        assert isinstance(self.test_increment, int)

        self.n_classes_per_task: int = self.increment
        self.test_increment = self.increment

    def apply(self, method: Method, config: Config = None) -> IncrementalSLResults:
        """Apply the given method on this setting to producing some results."""
        # TODO: It still isn't super clear what should be in charge of creating
        # the config, and how to create it, when it isn't passed explicitly.
        self.config = config or self._setup_config(method) 
        assert self.config

        method.configure(setting=self)

        # Run the main loop (which is defined in IncrementalAssumption).
        results: IncrementalSLResults = super().main_loop(method)
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

        # data_dir = data_dir or self.data_dir or self.config.data_dir
        # self.make_dataset(data_dir, download=True)
        # self.data_dir = data_dir
        return super().prepare_data(data_dir=data_dir, **kwargs)

    def setup(self, stage: str = None):
        super().setup(stage=stage)
        # TODO: Adding this temporarily just for the competition: The TestEnvironment
        # needs access to this information in order to split the metrics for each task.
        self.test_boundary_steps = [0] + list(
            itertools.accumulate(map(len, self.test_datasets))
        )[:-1]
        self.test_steps = sum(map(len, self.test_datasets))
        # self.test_steps = [0] + list(
        #     itertools.accumulate(map(len, self.test_datasets))
        # )[:-1]

    # def _make_train_dataset(self) -> Dataset:
    #     return self.train_datasets[self.current_task_id]

    # def _make_val_dataset(self) -> Dataset:
    #     return self.val_datasets[self.current_task_id]

    # def _make_test_dataset(self) -> Dataset:
    #     return concat(self.test_datasets)

    def train_dataloader(
        self, batch_size: int = None, num_workers: int = None
    ) -> IncrementalSLEnvironment:
        """ Returns a DataLoader for the train dataset of the current task. """
        train_env = super().train_dataloader(
            batch_size=batch_size, num_workers=num_workers
        )
        # Overwrite the wandb prefix for the `MeasureSLPerformanceWrapper` to include
        # the task id.
        if self.monitor_training_performance:
            # Overwrite the 'wandb prefix'
            assert isinstance(train_env, MeasureSLPerformanceWrapper)
            train_env.wandb_prefix = f"Train/Task {self.current_task_id}"
        self.train_env = train_env
        return self.train_env

    def val_dataloader(
        self, batch_size: int = None, num_workers: int = None
    ) -> PassiveEnvironment:
        """ Returns a DataLoader for the validation dataset of the current task. """
        val_env = super().val_dataloader(batch_size=batch_size, num_workers=num_workers)
        return self.val_env

    def test_dataloader(
        self, batch_size: int = None, num_workers: int = None
    ) -> PassiveEnvironment["ClassIncrementalSetting.Observations", Actions, Rewards]:
        """ Returns a DataLoader for the test dataset of the current task. """
        if not self.has_prepared_data:
            self.prepare_data()
        if not self.has_setup_test:
            self.setup("test")

        # Join all the test datasets.
        dataset = self._make_test_dataset()

        batch_size = batch_size if batch_size is not None else self.batch_size
        num_workers = num_workers if num_workers is not None else self.num_workers

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
        )

        # NOTE: The transforms from `self.transforms` (the 'base' transforms) were
        # already added when creating the datasets and the CL scenario.
        test_specific_transforms = self.additional_transforms(self.test_transforms)
        if test_specific_transforms:
            env = TransformObservation(env, f=test_specific_transforms)

        if self.config.device:
            # TODO: Put this before or after the image transforms?
            from sequoia.common.gym_wrappers.convert_tensors import ConvertToFromTensors
            env = ConvertToFromTensors(env, device=self.config.device)

        # TODO: Remove this, I don't think it's used anymore, since `hide_task_labels`
        # is an argument to self.Environment now.
        if not self.task_labels_at_test_time:
            env = HideTaskLabelsWrapper(env)

        # TODO: Remove this once that stuff with the 'fake' task schedule is fixed below,
        # base it on the equivalent in ContinualSLSetting instead (which should actually
        # be moved into DiscreteTaskAgnosticSL, now that I think about it!)

        # Testing this out, we're gonna have a "test schedule" like this to try
        # to imitate the MultiTaskEnvironment in RL.
        transition_steps = [0] + list(
            itertools.accumulate(map(len, self.test_datasets))
        )[:-1]
        # FIXME: Creating a 'task schedule' for the TestEnvironment, mimicing what's in
        # the RL settings.
        test_task_schedule = dict.fromkeys(
            [step // (env.batch_size or 1) for step in transition_steps],
            range(len(transition_steps)),
        )
        # TODO: Configure the 'monitoring' dir properly.
        if wandb.run:
            test_dir = wandb.run.dir
        else:
            test_dir = self.config.log_dir

        test_loop_max_steps = len(dataset) // (env.batch_size or 1)
        # TODO: Fix this: iteration doesn't ever end for some reason.

        test_env = IncrementalSLTestEnvironment(
            env,
            directory=test_dir,
            step_limit=test_loop_max_steps,
            task_schedule=test_task_schedule,
            force=True,
            config=self.config,
            video_callable=None if (wandb.run or self.config.render) else False,
        )

        if self.test_env:
            self.test_env.close()
        self.test_env = test_env
        return self.test_env

    def split_batch_function(
        self, training: bool
    ) -> Callable[[Tuple[Tensor, ...]], Tuple[Observations, Rewards]]:
        """ Returns a callable that is used to split a batch into observations and rewards.
        """
        assert False, "TODO: Removing this."
        task_classes = {
            i: self.task_classes(i, train=training) for i in range(self.nb_tasks)
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
            if self.shared_action_space:
                y = relabel(y, task_classes)

            if (training and not self.task_labels_at_train_time) or (
                not training and not self.task_labels_at_test_time
            ):
                # Remove the task labels if we're not currently allowed to have
                # them.
                # TODO: Using None might cause some issues. Maybe set -1 instead?
                t = None

            observations = self.Observations(x=x, task_labels=t)
            rewards = self.Rewards(y=y)

            return observations, rewards

        return split_batch

    def make_train_cl_scenario(self, train_dataset: _ContinuumDataset) -> _BaseScenario:
        """ Creates a train ClassIncremental object from continuum. """
        return ClassIncremental(
            train_dataset,
            nb_tasks=self.nb_tasks,
            increment=self.increment,
            initial_increment=self.initial_increment,
            class_order=self.class_order,
            transformations=self.transforms,
        )

    def make_test_cl_scenario(self, test_dataset: _ContinuumDataset) -> _BaseScenario:
        """ Creates a test ClassIncremental object from continuum. """
        return ClassIncremental(
            test_dataset,
            nb_tasks=self.nb_tasks,
            increment=self.test_increment,
            initial_increment=self.test_initial_increment,
            class_order=self.test_class_order,
            transformations=self.transforms,
        )

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

    def num_classes_in_current_task(self, train: bool = None) -> int:
        """ Returns the number of classes in the current task. """
        # TODO: Its ugly to have the 'method' tell us if we're currently in
        # train/eval/test, no? Maybe just make a method for each?
        return self.num_classes_in_task(self._current_task_id, train=train)

    def task_classes(self, task_id: int, train: bool) -> List[int]:
        """ Gives back the 'true' labels present in the given task. """
        start_index = sum(self.num_classes_in_task(i, train) for i in range(task_id))
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
        for loader_method in [
            self.train_dataloader,
            self.val_dataloader,
            self.test_dataloader,
        ]:
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
                assert x in self.observation_space["x"]

            for i in range(5):
                actions = env.action_space.sample()
                observations, rewards, done, info = env.step(actions)
                assert isinstance(observations, self.Observations), type(observations)
                assert isinstance(rewards, self.Rewards), type(rewards)
                actions = env.action_space.sample()
                if done:
                    observations = env.reset()
            env.close()


# def relabel(y: Tensor, task_classes: Dict[int, List[int]]) -> Tensor:
#     """ Relabel the elements of 'y' to their  index in the list of classes for
#     their task.

#     Example:

#     >>> import torch
#     >>> y = torch.as_tensor([2, 3, 2, 3, 2, 2])
#     >>> task_classes = {0: [0, 1], 1: [2, 3]}
#     >>> relabel(y, task_classes)
#     tensor([0, 1, 0, 1, 0, 0])
#     """
#     # TODO: Double-check that this never leaves any zeros where it shouldn't.
#     new_y = torch.zeros_like(y)
#     # assert unique_y <= set(task_classes), (unique_y, task_classes)
#     for task_id, task_true_classes in task_classes.items():
#         for i, label in enumerate(task_true_classes):
#             new_y[y == label] = i
#     return new_y


# This is just meant as a cleaner way to import the Observations/Actions/Rewards
# than particular setting.
Observations = IncrementalSLSetting.Observations
Actions = IncrementalSLSetting.Actions
Rewards = IncrementalSLSetting.Rewards

# TODO: I wouldn't want these above to overwrite / interfere with the import of
# the "base" versions of these objects from sequoia.settings.bases.objects, which are
# imported in settings/__init__.py. Will have to check that doing
# `from .passive import *` over there doesn't actually import these here.


if __name__ == "__main__":
    import doctest

    doctest.testmod()
