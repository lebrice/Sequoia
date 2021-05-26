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

from sequoia.common import ClassificationMetrics
from sequoia.common.config import Config
from sequoia.common.gym_wrappers import TransformObservation
from sequoia.common.gym_wrappers.batch_env.tile_images import tile_images
from sequoia.common.gym_wrappers.convert_tensors import add_tensor_support
from sequoia.common.gym_wrappers.utils import RenderEnvWrapper
from sequoia.common.spaces import Image, Sparse
from sequoia.common.spaces.named_tuple import NamedTupleSpace
from sequoia.common.transforms import Transforms
from sequoia.settings.assumptions.incremental import (
    IncrementalAssumption,
    TaskResults,
    TaskSequenceResults,
    IncrementalResults,
    TestEnvironment,
)
from sequoia.settings.base import Method, Results
from sequoia.utils import get_logger

# TODO: Fix this: not sure where this 'SLSetting' should be.
from sequoia.settings.sl.environment import Actions, PassiveEnvironment, Rewards
from sequoia.settings.sl.setting import SLSetting
from sequoia.settings.sl.continual import ContinualSLSetting
from sequoia.settings.sl.wrappers import MeasureSLPerformanceWrapper
from sequoia.settings.rl.wrappers import HideTaskLabelsWrapper
from continuum.tasks import concat


from ..discrete.setting import DiscreteTaskAgnosticSLSetting
from .results import IncrementalSLResults
from .environment import IncrementalSLEnvironment
from .objects import Observations, ObservationType, Actions, ActionType, Rewards, RewardType
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

    # TODO: Need to put num_workers in only one place.
    batch_size: int = field(default=32, cmd=False)
    num_workers: int = field(default=4, cmd=False)

    # Wether or not to relabel the y's to be within the [0, n_classes_per_task]
    # range. Floating (False by default) in Class-Incremental Setting, but set to True
    # in domain_incremental Setting.
    shared_action_space: bool = False

    # TODO: IDEA: Adding these fields/constructor arguments so that people can pass a
    # custom ready-made `Scenario` from continuum to use.
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

    def apply(self, method: Method, config: Config = None) -> IncrementalSLResults:
        """Apply the given method on this setting to producing some results."""
        # TODO: It still isn't super clear what should be in charge of creating
        # the config, and how to create it, when it isn't passed explicitly.
        if config is not None:
            self.config = config
            logger.debug(f"Using Config {self.config}")
        elif isinstance(getattr(method, "config", None), Config):
            # If the Method has a `config` attribute that is a Config, use that.
            self.config = getattr(method, "config")
            logger.debug(f"Using Config from the Method: {self.config}")
        else:
            logger.debug("Parsing the Config from the command-line.")
            self.config = Config.from_args(self._argv, strict=False)
            logger.debug(f"Resulting Config: {self.config}")
        assert self.config is not None

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

        data_dir = data_dir or self.data_dir or self.config.data_dir
        self.make_dataset(data_dir, download=True)
        self.data_dir = data_dir
        super().prepare_data(**kwargs)

    def setup(self, stage: str=None):
        super().setup(stage=stage)
        # TODO: Adding this temporarily just for the competition
        self.test_boundary_steps = [0] + list(
            itertools.accumulate(map(len, self.test_datasets))
        )[:-1]
        self.test_steps = sum(map(len, self.test_datasets))
        # self.test_steps = [0] + list(
        #     itertools.accumulate(map(len, self.test_datasets))
        # )[:-1]

    def _make_train_dataset(self) -> Dataset:
        return self.train_datasets[self.current_task_id]

    def _make_val_dataset(self) -> Dataset:
        return self.val_datasets[self.current_task_id]

    def _make_test_dataset(self) -> Dataset:
        return concat(self.test_datasets)

    def train_dataloader(
        self, batch_size: int = None, num_workers: int = None
    ) -> IncrementalSLEnvironment:
        """ Returns a DataLoader for the train dataset of the current task. """
        train_env = super().train_dataloader(batch_size=batch_size, num_workers=num_workers)
        # TODO: Set a different prefix for `MeasureSLPerformanceWrapper`
        if self.monitor_training_performance:
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
            test_dir = "results"

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
                assert x in self.observation_space[0]

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


class IncrementalSLTestEnvironment(TestEnvironment):
    def __init__(
        self, env: gym.Env, *args, task_schedule: Dict[int, Any] = None, **kwargs
    ):
        super().__init__(env, *args, **kwargs)
        self._steps = 0
        # TODO: Maybe rework this so we don't depend on the test phase being one task at
        # a time, instead store the test metrics in the task corresponding to the
        # task_label in the observations.
        # BUG: The problem is, right now we're depending on being passed the
        # 'task schedule', which we then use to get the task ids. This
        # is actually pretty bad, because if the class ordering was changed between
        # training and testing, then, this wouldn't actually report the correct results!
        self.task_schedule = task_schedule or {}
        self.task_steps = sorted(self.task_schedule.keys())
        self.results: TaskSequenceResults[ClassificationMetrics] = TaskSequenceResults(
            TaskResults() for step in self.task_steps
        )
        self._reset = False
        # NOTE: The task schedule is already in terms of the number of batches.
        self.boundary_steps = [step for step in self.task_schedule.keys()]

    def get_results(self) -> IncrementalSLResults:
        return self.results

    def reset(self):
        if not self._reset:
            logger.debug("Initial reset.")
            self._reset = True
            return super().reset()
        else:
            logger.debug("Resetting the env closes it.")
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

        # Given the step, find the task id.
        task_id = bisect.bisect_right(task_steps, self._steps) - 1
        self.results[task_id].append(metric)
        self._steps += 1

        # Debugging issue with Monitor class:
        # return super()._after_step(observation, reward, done, info)
        if not self.enabled:
            return done

        if done and self.env_semantics_autoreset:
            # For envs with BlockingReset wrapping VNCEnv, this observation will be the
            # first one of the new episode
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

    def _after_reset(self, observation: IncrementalSLSetting.Observations):
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

        # -- Code from Monitor
        if not self.enabled:
            return
        # Reset the stat count
        self.stats_recorder.after_reset(observation)
        if self.config.render:
            self.reset_video_recorder()

        # Bump *after* all reset activity has finished
        self.episode_id += 1

        self._flush()
        # --

    def render(self, mode="human", **kwargs):
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
