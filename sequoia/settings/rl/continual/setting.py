""" Current most general Setting in the Reinforcement Learning side of the tree.
"""
import difflib
import itertools
import textwrap
import json
import math
import warnings
from copy import deepcopy
from dataclasses import dataclass, fields
from functools import partial
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, List, Optional, Sequence, Type, Union

import gym
import numpy as np
import wandb
from gym import spaces
from gym.envs.classic_control import CartPoleEnv, MountainCarEnv, PendulumEnv
from gym.envs.registration import EnvRegistry, EnvSpec, load, registry, spec
from gym.utils import colorize
from gym.wrappers import TimeLimit
from simple_parsing import choice, field, list_field
from simple_parsing.helpers import dict_field
from stable_baselines3.common.atari_wrappers import AtariWrapper

from sequoia.common import Config
from sequoia.common.gym_wrappers import (
    AddDoneToObservation,
    MultiTaskEnvironment,
    RenderEnvWrapper,
    SmoothTransitions,
    TransformObservation,
)
from sequoia.utils.utils import pairwise, deprecated_property
from sequoia.common.gym_wrappers.action_limit import ActionLimit
from sequoia.common.gym_wrappers.batch_env.tile_images import tile_images
from sequoia.common.gym_wrappers.convert_tensors import add_tensor_support
from sequoia.common.gym_wrappers.env_dataset import EnvDataset
from sequoia.common.gym_wrappers.episode_limit import EpisodeLimit
from sequoia.common.gym_wrappers.pixel_observation import (
    ImageObservations,
    PixelObservationWrapper,
)
from sequoia.common.gym_wrappers.utils import (
    IterableWrapper,
    is_atari_env,
    is_classic_control_env,
    is_monsterkong_env,
)
from sequoia.common.metrics.rl_metrics import EpisodeMetrics
from sequoia.common.spaces import Sparse, TypedDictSpace
from sequoia.common.transforms import Transforms
from sequoia.settings.assumptions.continual import (
    ContinualAssumption,
    ContinualResults,
    TestEnvironment,
)
from sequoia.settings.base import Method, Results
from sequoia.settings.rl import ActiveEnvironment, RLSetting
from sequoia.settings.rl.wrappers import (
    HideTaskLabelsWrapper,
    MeasureRLPerformanceWrapper,
    TypedObjectsWrapper,
)
from sequoia.utils import get_logger
from sequoia.utils.utils import camel_case, flag


from .environment import GymDataLoader
from .make_env import make_batched_env
from .objects import (
    Actions,
    ActionType,
    Observations,
    ObservationType,
    Rewards,
    RewardType,
)
from .results import ContinualRLResults
from .tasks import (
    ContinuousTask,
    TaskSchedule,
    is_supported,
    make_continuous_task,
    names_match,
)
from .test_environment import ContinualRLTestEnvironment

logger = get_logger(__file__)


# Type alias for the Environment returned by `train/val/test_dataloader`.
Environment = ActiveEnvironment[
    "ContinualRLSetting.Observations",
    "ContinualRLSetting.Observations",
    "ContinualRLSetting.Rewards",
]


# NOTE: Takes about 0.2 seconds to check for all compatible envs (with loading), and
# only happens once.
supported_envs: Dict[str, EnvSpec] = {
    spec.id: spec for env_id, spec in registry.env_specs.items() if is_supported(env_id)
}
available_datasets: Dict[str, str] = {env_id: env_id for env_id in supported_envs}
# available_datasets.update(
#     {camel_case(env_id.split("-v")[0]): env_id for env_id in supported_envs}
# )


@dataclass
class ContinualRLSetting(RLSetting, ContinualAssumption):
    """ Reinforcement Learning Setting where the environment changes over time.

    This is an Active setting which uses gym environments as sources of data.
    These environments' attributes could change over time following a task
    schedule. An example of this could be that the gravity increases over time
    in cartpole, making the task progressively harder as the agent interacts with
    the environment.
    """

    # (NOTE: commenting out SLSetting.Observations as it is the same class
    # as Setting.Observations, and we want a consistent method resolution order.
    Observations: ClassVar[Type[Observations]] = Observations
    Actions: ClassVar[Type[Actions]] = Actions
    Rewards: ClassVar[Type[Rewards]] = Rewards

    # The type of results returned by an RL experiment.
    Results: ClassVar[Type[Results]] = ContinualRLResults
    # The type wrapper used to wrap the test environment, and which produces the
    # results.
    TestEnvironment: ClassVar[Type[TestEnvironment]] = ContinualRLTestEnvironment

    # Dict of all available options for the 'dataset' field below.
    available_datasets: ClassVar[Dict[str, Union[str, Any]]] = available_datasets
    # The function used to create the tasks for the chosen env.
    _task_sampling_function: ClassVar[
        Callable[..., ContinuousTask]
    ] = make_continuous_task

    # Which environment (a.k.a. "dataset") to learn on.
    # The dataset could be either a string (env id or a key from the
    # available_datasets dict), a gym.Env, or a callable that returns a
    # single environment.
    dataset: str = choice(available_datasets, default="CartPole-v0")

    # The number of "tasks" that will be created for the training, valid and test
    # environments.
    # NOTE: In the case of settings with smooth task boundaries, this is the number of
    # "base" tasks which are created, and the task space consists of interpolations
    # between these base tasks.
    # When left unset, will use a default value that makes sense
    # (something like 5).
    nb_tasks: int = field(5, alias=["n_tasks", "num_tasks"])

    # Environment/dataset to use for validation. Defaults to the same as `dataset`.
    train_dataset: Optional[str] = None
    # Environment/dataset to use for validation. Defaults to the same as `dataset`.
    val_dataset: Optional[str] = None
    # Environment/dataset to use for testing. Defaults to the same as `dataset`.
    test_dataset: Optional[str] = None

    # Wether the task boundaries are smooth or sudden.
    smooth_task_boundaries: bool = True
    # Wether the tasks are sampled uniformly. (This is set to True in MultiTaskRLSetting
    # and below)
    stationary_context: bool = False

    # Max number of training steps in total. (Also acts as the "length" of the training
    # and validation "Datasets")
    train_max_steps: int = 100_000
    # Maximum number of episodes in total.
    # TODO: Add tests for this 'max episodes' and 'episodes_per_task'.
    train_max_episodes: Optional[int] = None
    # Total number of steps in the test loop. (Also acts as the "length" of the testing
    # environment.)
    test_max_steps: int = 10_000
    test_max_episodes: Optional[int] = None
    # Standard deviation of the multiplicative Gaussian noise that is used to
    # create the values of the env attributes for each task.
    task_noise_std: float = 0.2
    # NOTE: THIS ARG IS DEPRECATED! Only keeping it here so previous config yaml files
    # don't cause a crash.
    observe_state_directly: Optional[bool] = None

    # NOTE: Removing those, in favor of just using the registered Pixel<...>-v? variant.
    # force_pixel_observations: bool = False
    # """ Wether to use the "pixel" version of `self.dataset`.
    # When `False`, does nothing.
    # When `True`, will do one of the following, depending on the choice of environment:
    # - For classic control envs, it adds a `PixelObservationsWrapper` to the env.
    # - For atari envs:
    #     - If `self.dataset` is a regular atari env (e.g. "Breakout-v0"), does nothing.
    #     - if `self.dataset` is the 'RAM' version of an atari env, raises an error.
    # - For mujoco envs, this raises a NotImplementedError, as we don't yet know how to
    #   make a pixel-version the Mujoco Envs.
    # - For other envs:
    #     - If the environment's observation space appears to be image-based, an error
    #       will be raised.
    #     - If the environment's observation space doesn't seem to be image-based, does
    #       nothing.
    # """

    # force_state_observations: bool = False
    # """ Wether to use the "state" version of `self.dataset`.
    # When `False`, does nothing.
    # When `True`, will do one of the following, depending on the choice of environment:
    # - For classic control envs, it does nothing, as they are already state-based.
    # - TODO: For atari envs, the 'RAM' version of the chosen env will be used.
    # - For mujoco envs, it doesn nothing, as they are already state-based.
    # - For other envs, if this is set to True, then
    #     - If the environment's observation space appears to be image-based, an error
    #       will be raised.
    #     - If the environment's observation space doesn't seem to be image-based, does
    #       nothing.
    # """

    # NOTE: Removing this from the continual setting.
    # By default 1 for this setting, meaning that the context is a linear interpolation
    # between the start context (usually the default task for the environment) and a
    # randomly sampled task.
    # nb_tasks: int = field(5, alias=["n_tasks", "num_tasks"])

    # Wether to convert the observations / actions / rewards of the envs (and their
    # spaces) such that they return Tensors rather than numpy arrays.
    # TODO: Maybe switch this to True by default?
    prefer_tensors: bool = False

    # Path to a json file from which to read the train task schedule.
    train_task_schedule_path: Optional[Path] = None
    # Path to a json file from which to read the validation task schedule.
    val_task_schedule_path: Optional[Path] = None
    # Path to a json file from which to read the test task schedule.
    test_task_schedule_path: Optional[Path] = None

    # Wether observations from the environments whould include
    # the end-of-episode signal. Only really useful if your method will iterate
    # over the environments in the dataloader style
    # (as does the baseline method).
    add_done_to_observations: bool = False

    # The maximum number of steps per episode. When None, there is no limit.
    max_episode_steps: Optional[int] = None

    # Transforms to be applied by default to the observatons of the train/valid/test
    # environments.
    transforms: List[Transforms] = list_field()
    # Transforms to be applied to the training environment, in addition to those already
    # in `transforms`.
    train_transforms: List[Transforms] = list_field()
    # Transforms to be applied to the validation environment, in addition to those
    # already in `transforms`.
    val_transforms: List[Transforms] = list_field()
    # Transforms to be applied to the testing environment, in addition to those already
    # in `transforms`.
    test_transforms: List[Transforms] = list_field()

    # When True, a Monitor-like wrapper will be applied to the training environment
    # and monitor the 'online' performance during training. Note that in SL, this will
    # also cause the Rewards (y) to be withheld until actions are passed to the `send`
    # method of the Environment.
    monitor_training_performance: bool = flag(True)

    #
    # -------- Fields below don't have corresponding command-line arguments. -----------
    #
    train_task_schedule: Dict[int, Dict[str, float]] = dict_field(cmd=False)
    val_task_schedule: Dict[int, Dict[str, float]] = dict_field(cmd=False)
    test_task_schedule: Dict[int, Dict[str, float]] = dict_field(cmd=False)

    # TODO: Naming is a bit inconsistent, using `valid` here, whereas we use `val`
    # elsewhere.
    train_wrappers: List[Callable[[gym.Env], gym.Env]] = list_field(cmd=False)
    val_wrappers: List[Callable[[gym.Env], gym.Env]] = list_field(cmd=False)
    test_wrappers: List[Callable[[gym.Env], gym.Env]] = list_field(cmd=False)

    # keyword arguments to be passed to the base environment through gym.make(base_env, **kwargs).
    base_env_kwargs: Dict = dict_field(cmd=False)

    batch_size: Optional[int] = field(default=None, cmd=False)
    num_workers: Optional[int] = field(default=None, cmd=False)

    # Maximum number of training steps per task.
    # NOTE: In this particular setting, since there aren't clear 'tasks' to speak of, we
    # don't expose this option on the command-line.
    train_steps_per_task: Optional[int] = field(default=None, to_dict=False, cmd=False)
    # Number of test steps per task.
    # NOTE: In this particular setting, since there aren't clear 'tasks' to speak of, we
    # don't expose this option on the command-line.
    test_steps_per_task: Optional[int] = field(default=None, to_dict=False, cmd=False)

    # Deprecated: use `train_max_steps` instead.
    max_steps: Optional[int] = deprecated_property("max_steps", "train_max_steps")
    # Deprecated: use `train_max_steps` instead.
    test_steps: Optional[int] = deprecated_property("test_steps", "test_max_steps")
    # Deprecated, use `train_steps_per_task` instead.
    steps_per_task: Optional[int] = field(default=None, to_dict=False, cmd=False)

    def __post_init__(self):
        super().__post_init__()

        # TODO: Fix nnoying little issues with this trio of fields that are interlinked:
        if self.test_steps_per_task is not None:
            if self.test_max_steps == 10_000:
                self.test_max_steps = self.nb_tasks * self.test_steps_per_task
            else:
                self.nb_tasks = self.test_max_steps // self.test_steps_per_task

        renamed_fields = {
            # "max_steps": "train_max_steps",
            # "test_steps": "test_max_steps",
            "steps_per_task": "train_steps_per_task",
            # "steps_per_task": "train_steps_per_task",
        }
        for old_field_name, new_field_name in renamed_fields.items():
            passed_value = getattr(self, old_field_name)
            if passed_value is not None:
                warnings.warn(
                    DeprecationWarning(
                        f"Field '{old_field_name}' has been renamed to '{new_field_name}'."
                    )
                )
                setattr(self, new_field_name, passed_value)
        # if self.max_steps is not None:
        #     warnings.warn(DeprecationWarning("'max_steps' is deprecated, use 'train_max_steps' instead."))
        #     self.train_max_steps = self.max_steps
        # if self.test_steps is not None:
        #     warnings.warn(DeprecationWarning("'test_steps' is deprecated, use 'test_max_steps' instead."))     

        if (
            self.dataset not in self.available_datasets.values()
        ):
            try:
                self.dataset = find_matching_dataset(
                    self.available_datasets, self.dataset
                )
            except NotImplementedError as e:
                # FIXME: Removing this warning in the case where a custom env is pased
                # for each task. However, the train_envs field is only created in a
                # subclass, so this check is ugly.
                if not (hasattr(self, "train_envs") and self.dataset is self.train_envs[0]):
                    warnings.warn(
                        RuntimeWarning(
                            f"Will attempt to use unsupported dataset {textwrap.shorten(str(self.dataset), 100)}!"
                        )
                    )
            except Exception as e:
                raise gym.error.UnregisteredEnv(
                    f"({e}) The chosen dataset/environment ({self.dataset}) isn't in the dict of "
                    f"available datasets/environments, and a task schedule was not passed, "
                    f"so this Setting ({type(self).__name__}) doesn't know how to create "
                    f"tasks for that env!\n"
                    f"Supported envs:\n"
                    + ("\n".join(f"- {k}: {v}" for k, v in self.available_datasets.items()))
                )
        logger.info(f"Chosen dataset: {textwrap.shorten(str(self.dataset), 50)}")

        # The ids of the train/valid/test environments.
        self.train_dataset: Union[str, Callable[[], gym.Env]] = self.train_dataset or self.dataset
        self.val_dataset: Union[str, Callable[[], gym.Env]] = self.val_dataset or self.dataset
        self.test_dataset: Union[str, Callable[[], gym.Env]] = self.test_dataset or self.dataset

        # # The environment 'ID' associated with each 'simple name'.
        # self.train_dataset_id: str = self._get_dataset_id(self.train_dataset)
        # self.val_dataset_id: str = self._get_dataset_id(self.val_dataset)
        # self.train_dataset_id: str = self._get_dataset_id(self.train_dataset)

        # Set the number of tasks depending on the increment, and vice-versa.
        # (as only one of the two should be used).
        assert self.train_max_steps, "assuming this should always be set, for now."

        # Load the task schedules from the corresponding files, if present.
        if self.train_task_schedule_path:
            self.train_task_schedule = _load_task_schedule(
                self.train_task_schedule_path
            )
            self.nb_tasks = len(self.train_task_schedule) - 1
        if self.val_task_schedule_path:
            self.val_task_schedule = _load_task_schedule(self.val_task_schedule_path)
        if self.test_task_schedule_path:
            self.test_task_schedule = _load_task_schedule(self.test_task_schedule_path)

        self.train_env: gym.Env
        self.valid_env: gym.Env
        self.test_env: gym.Env

        # Temporary environments which are created and used only for creating the task
        # schedules and the 'base' observation spaces, and then closed right after.
        self._temp_train_env: Optional[gym.Env] = self._make_env(self.train_dataset)
        self._temp_val_env: Optional[gym.Env] = None
        self._temp_test_env: Optional[gym.Env] = None
        # Create the task schedules, using the 'task sampling' function from `tasks.py`.

        if not self.train_task_schedule:
            self.train_task_schedule = self.create_train_task_schedule()
        else:
            if max(self.train_task_schedule) == len(self.train_task_schedule) - 1:
                # If the keys correspond to the task ids rather than the transition steps:
                nb_tasks = len(self.train_task_schedule) - 1
                if self.train_steps_per_task is not None:
                    steps_per_task = self.train_steps_per_task
                else:
                    steps_per_task = self.train_max_steps // nb_tasks
                self.train_task_schedule = type(self.train_task_schedule)(
                    {
                        i * steps_per_task: self.train_task_schedule[step]
                        for i, step in enumerate(
                            sorted(self.train_task_schedule.keys())
                        )
                    }
                )
            nb_tasks = len(self.train_task_schedule) - 1
            logger.info(
                f"Setting the number of tasks to {nb_tasks} based on the task schedule."
            )
            self.nb_tasks = nb_tasks
            # self.train_max_steps = max(self.train_task_schedule)

        if not self.val_task_schedule:
            # Avoid creating an additional env, just reuse the train_temp_env.
            self._temp_val_env = (
                self._temp_train_env
                if self.val_dataset == self.train_dataset
                else self._make_env(self.val_dataset)
            )
            self.val_task_schedule = self.create_val_task_schedule()
        elif max(self.val_task_schedule) == len(self.val_task_schedule) - 1:
            # If the keys correspond to the task ids rather than the transition steps
            nb_tasks = len(self.val_task_schedule)
            steps_per_task = self.train_max_steps // nb_tasks
            self.val_task_schedule = type(self.val_task_schedule)(
                **{
                    i * steps_per_task: self.val_task_schedule[step]
                    for i, step in enumerate(sorted(self.val_task_schedule.keys()))
                }
            )

        if not self.test_task_schedule:
            self._temp_test_env = (
                self._temp_train_env
                if self.test_dataset == self.train_dataset
                else self._make_env(self.val_dataset)
            )
            self.test_task_schedule = self.create_test_task_schedule()
        else:
            if max(self.test_task_schedule) == len(self.test_task_schedule) - 1:
                # If the keys correspond to the task ids rather than the transition steps
                nb_tasks = len(self.test_task_schedule)
                if self.test_steps_per_task is not None:
                    steps_per_task = self.test_steps_per_task
                else:
                    steps_per_task = self.test_max_steps // nb_tasks
                self.test_task_schedule = type(self.test_task_schedule)(
                    **{
                        i * steps_per_task: self.test_task_schedule[step]
                        for i, step in enumerate(sorted(self.test_task_schedule.keys()))
                    }
                )
            # self.test_max_steps = max(self.test_task_schedule)

        if self._temp_train_env:
            self._temp_train_env.close()
        if self._temp_val_env and self._temp_val_env is not self._temp_train_env:
            self._temp_val_env.close()
        if self._temp_test_env and self._temp_test_env is not self._temp_train_env:
            self._temp_test_env.close()

        defaults = {f.name: f.default for f in fields(self)}
        train_task_lengths: List[int] = [
            task_b_step - task_a_step
            for task_a_step, task_b_step in pairwise(
                sorted(self.train_task_schedule.keys())
            )
        ]
        # TODO: This will crash if nb_tasks is 1, right?
        # train_max_steps = train_last_boundary + train_task_lengths[-1]
        test_task_lengths: List[int] = [
            task_b_step - task_a_step
            for task_a_step, task_b_step in pairwise(
                sorted(self.test_task_schedule.keys())
            )
        ]
        if 0 not in self.train_task_schedule.keys():
            raise RuntimeError(
                "`train_task_schedule` needs an entry at key 0, as the initial state"
            )
        if 0 not in self.test_task_schedule.keys():
            raise RuntimeError(
                "`test_task_schedule` needs an entry at key 0, as the initial state"
            )
        if self.train_max_steps != max(self.train_task_schedule):
            if self.train_max_steps == defaults["train_max_steps"]:
                self.train_max_steps = max(self.train_task_schedule)
                logger.info(f"Setting `train_max_steps` to {self.train_max_steps}")
            else:
                raise RuntimeError(
                    f"For now, the train task schedule needs to have a value at key "
                    f"`train_max_steps` ({self.train_max_steps})."
                )
        if self.test_max_steps != max(self.test_task_schedule):
            if self.test_max_steps == defaults["test_max_steps"]:
                logger.info(f"Setting `test_max_steps` to {self.train_max_steps}")
                self.test_max_steps = max(self.test_task_schedule)

        if self.test_max_steps != max(self.test_task_schedule):
            raise RuntimeError(
                f"For now, the test task schedule needs to have a value at key "
                f"`test_max_steps` ({self.test_max_steps}). "
            )
        if not (
            len(self.train_task_schedule)
            == len(self.test_task_schedule)
            == len(self.val_task_schedule)
        ):
            raise RuntimeError(
                "Training, validation and testing task schedules should have the same "
                "number of items for now."
            )

        # Expected value for self.nb_tasks
        nb_tasks = len(self.train_task_schedule) - 1
        # if self.nb_tasks != nb_tasks:
        #     raise RuntimeError(
        #         f"Expected `nb_tasks` to be {nb_tasks}, since there are "
        #         f"{len(train_task_schedule)} tasks in the task schedule, but got value "
        #         f"of {self.nb_tasks} instead!"
        #     )

        train_last_boundary = max(
            set(self.train_task_schedule.keys()) - {self.train_max_steps}
        )
        test_last_boundary = max(
            set(self.test_task_schedule.keys()) - {self.test_max_steps}
        )
        if self.nb_tasks != nb_tasks:
            if self.nb_tasks == defaults["nb_tasks"]:
                assert len(self.train_task_schedule) == len(self.test_task_schedule)
                self.nb_tasks = len(self.train_task_schedule) - 1
                logger.info(
                    f"`nb_tasks` set to {self.nb_tasks} based on the task schedule"
                )
            else:
                raise RuntimeError(
                    f"The passed number of tasks ({self.nb_tasks}) is inconsistent "
                    f"with the passed task schedules, which have {nb_tasks} tasks."
                )

        if not train_task_lengths:
            assert not test_task_lengths
            assert nb_tasks == 1
            assert self.train_max_steps > 0
            assert self.test_max_steps > 0
            train_max_steps = self.train_max_steps
            test_max_steps = self.test_max_steps
        else:
            train_max_steps = sum(train_task_lengths)
            test_max_steps = sum(test_task_lengths)
            # train_max_steps = round(train_last_boundary + train_task_lengths[-1])
            # test_max_steps = round(test_last_boundary + test_task_lengths[-1])

        if self.train_max_steps != train_max_steps:
            if self.train_max_steps == defaults["train_max_steps"]:
                self.train_max_steps = train_max_steps
            else:
                raise RuntimeError(
                    f"Value of train_max_steps ({self.train_max_steps}) is "
                    f"inconsistent with the given train task schedule, which has "
                    f"the last task boundary at step {train_last_boundary}, with "
                    f"task lengths of {train_task_lengths}, as it suggests the maximum "
                    f"total number of steps to be {train_last_boundary} + "
                    f"{train_task_lengths[-1]} => {train_max_steps}!"
                )
        if self.test_max_steps != test_max_steps:
            if self.test_max_steps == defaults["test_max_steps"]:
                self.test_max_steps = test_max_steps
            else:
                raise RuntimeError(
                    f"Value of test_max_steps ({self.test_max_steps}) is "
                    f"inconsistent with the given tet task schedule, which has "
                    f"the last task boundary at step {test_last_boundary}, with "
                    f"task lengths of {test_task_lengths}, as it suggests the maximum "
                    f"total number of steps to be {test_last_boundary} + "
                    f"{test_task_lengths[-1]} => {test_max_steps}!"
                )

    def create_train_task_schedule(self) -> TaskSchedule:
        # change_steps = [0, self.train_max_steps]
        # Ex: nb_tasks == 5, train_max_steps = 10_000:
        # change_steps = [0, 2_000, 4_000, 6_000, 8_000, 10_000]
        if self.train_steps_per_task is not None:
            # TODO: a bit ugly, essentially need to check if this is for the  for subclasses, only for this setting.
            if self.smooth_task_boundaries:
                train_max_steps = self.train_steps_per_task * (self.nb_tasks + 1)
            else:
                train_max_steps = self.train_steps_per_task * self.nb_tasks
        else:
            train_max_steps = self.train_max_steps
        task_schedule_keys = np.linspace(
            0, train_max_steps, self.nb_tasks + 1, endpoint=True, dtype=int
        ).tolist()
        return self.create_task_schedule(
            temp_env=self._temp_train_env,
            change_steps=task_schedule_keys,
            # # TODO: Add properties for the train/valid/test seeds?
            # seed=self.train_seed,
        )

    def create_val_task_schedule(self) -> TaskSchedule:
        # Always the same as train task schedule for now.
        return self.train_task_schedule.copy()

    def create_test_task_schedule(self) -> TaskSchedule[ContinuousTask]:
        # Re-scale the steps in the task schedule based on self.test_max_steps
        # NOTE: Using the same task schedule as in training and validation for now.
        if self.train_task_schedule:
            nb_tasks = len(self.train_task_schedule) - 1
        else:
            nb_tasks = self.nb_tasks
        # TODO: Do we want to re-allow the `test_steps_per_task` argument?
        if self.test_steps_per_task is not None:
            test_max_steps = self.test_steps_per_task * nb_tasks
        else:
            test_max_steps = self.test_max_steps
        test_task_schedule_keys = np.linspace(
            0, test_max_steps, nb_tasks + 1, endpoint=True, dtype=int
        ).tolist()
        return {
            step: task
            for step, task in zip(
                test_task_schedule_keys, self.train_task_schedule.values()
            )
        }

    def create_task_schedule(
        self, temp_env: gym.Env, change_steps: List[int], seed: int = None,
    ) -> Dict[int, Dict]:
        """ Create the task schedule, which maps from a step to the changes that
        will occur in the environment when that step is reached.

        Uses the provided `temp_env` to generate the random tasks at the steps
        given in `change_steps` (a list of integers).

        Returns a dictionary mapping from integers (the steps) to the changes
        that will occur in the env at that step.

        TODO: For now in ContinualRL we use an interpolation of a dict of attributes
        to be set on the unwrapped env, but in IncrementalRL it is possible to pass
        callables to be applied on the environment at a given timestep.
        """
        task_schedule: Dict[int, Dict] = {}
        # TODO: Make it possible to use something other than steps as keys in the task
        # schedule, something like a NamedTuple[int, DeltaType], e.g. Episodes(10) or Steps(10)
        # something like that!
        # IDEA: Even fancier, we could use a TimeDelta to say "do one hour of task 0"!!
        for step in change_steps:
            # TODO: Pass wether its for training/validation/testing?
            task = type(self)._task_sampling_function(
                temp_env,
                step=step,
                change_steps=change_steps,
                seed=self.config.seed if self.config else None,
            )
            task_schedule[step] = task

        return task_schedule

    @property
    def observation_space(self) -> TypedDictSpace:
        """ The un-batched observation space, based on the choice of dataset and
        the transforms at `self.transforms` (which apply to the train/valid/test
        environments).

        The returned spaces is a TypedDictSpace, with the following properties/items:
        - `x`: observation space (e.g. `Image` space)
        - `task_labels`: Union[Discrete, Sparse[Discrete]]
           The task labels for each sample when task labels are available,
           otherwise the task labels space is `Sparse`, and entries will be `None`.
        """
        # TODO: Is it right that we set the observation space on the Setting to be the
        # observation space of the current train environment? 
        # In what situation could there be any difference between those?
        # - Changing the 'transforms' attributes after training?
        # if self.train_env is not None:
        #     # assert self._observation_space == self.train_env.observation_space
        #     return self.train_env.observation_space

        x_space = self._temp_train_env.observation_space
        # apply the transforms to the observation space.
        for transform in self.transforms:
            x_space = transform(x_space)

        task_label_space = self.task_label_space

        done_space = spaces.Box(0, 1, shape=(), dtype=bool)
        if not self.add_done_to_observations:
            done_space = Sparse(done_space, sparsity=1)

        observation_space = TypedDictSpace(
            x=x_space,
            task_labels=task_label_space,
            done=done_space,
            dtype=self.Observations,
        )

        if self.prefer_tensors:
            observation_space = add_tensor_support(observation_space)
        assert isinstance(observation_space, TypedDictSpace)

        if self.train_env is not None:
            # FIXME: Remove this perhaps. Just making sure that the Setting's
            # observation space is consistent with that of its environments.
            # NOTE: This check is a bit too strict, the task label space's sparsity for
            # instance isn't exactly the same.
            # assert observation_space == self.train_env.observation_space
            assert observation_space.x == self.train_env.observation_space.x, (observation_space, self.train_env.observation_space)
            # assert observation_space.task_labels.n == self.train_env.observation_space.task_labels.n
        return observation_space

    @property
    def task_label_space(self) -> gym.Space:
        # TODO: Explore an alternative design for the task sampling, based more around
        # gym spaces rather than the generic function approach that's currently used?
        # FIXME: This isn't really elegant, there isn't a `nb_tasks` attribute on the
        # ContinualRLSetting anymore, so we have to do a bit of a hack.. Would be
        # cleaner to maybe put this in the assumption class, under
        # `self.task_label_space`?
        task_label_space = spaces.Box(0.0, 1.0, shape=())
        if not self.task_labels_at_train_time or not self.task_labels_at_test_time:
            sparsity = 1
            if self.task_labels_at_train_time ^ self.task_labels_at_test_time:
                # We have task labels "50%" of the time, ish:
                sparsity = 0.5
            task_label_space = Sparse(task_label_space, sparsity=sparsity)
        return task_label_space

    @property
    def action_space(self) -> gym.Space:
        # TODO: Convert the action/reward spaces so they also use TypedDictSpace (even
        # if they just have one item), so that it correctly reflects the objects that
        # the envs accept.
        y_pred_space = self._temp_train_env.action_space
        # action_space = TypedDictSpace(y_pred=y_pred_space, dtype=self.Actions)
        return y_pred_space

    @property
    def reward_space(self) -> gym.Space:
        reward_range = self._temp_train_env.reward_range
        return getattr(
            self._temp_train_env,
            "reward_space",
            spaces.Box(reward_range[0], reward_range[1], shape=()),
        )

    def apply(
        self, method: Method, config: Config = None
    ) -> "ContinualRLSetting.Results":
        """Apply the given method on this setting to producing some results. """
        # Use the supplied config, or parse one from the arguments that were
        # used to create `self`.
        self.config: Config = config or self._setup_config(method)
        logger.debug(f"Config: {self.config}")

        # TODO: Test to make sure that this doesn't cause any other bugs with respect to
        # the display of stuff:
        # Call this method, which creates a virtual display if necessary.
        self.config.get_display()

        # TODO: Should we really overwrite the method's 'config' attribute here?
        if not getattr(method, "config", None):
            method.config = self.config

        # TODO: Remove `Setting.configure(method)` entirely, from everywhere,
        # and use the `prepare_data` or `setup` methods instead (since these
        # `configure` methods aren't using the `method` anyway.)
        method.configure(setting=self)

        # BUG This won't work if the task schedule uses callables as the values (as
        # they aren't json-serializable.)
        if self.stationary_context:
            logger.info(
                "Train tasks: "
                + json.dumps(list(self.train_task_schedule.values()), indent="\t")
            )
        else:
            try:
                logger.info(
                    "Train task schedule:"
                    + json.dumps(self.train_task_schedule, indent="\t")
                )
                # BUG: Sometimes the task schedule isnt json-serializable!
            except TypeError:
                logger.info("Train task schedule: ")
                for key, value in self.train_task_schedule.items():
                    logger.info(f"{key}: {value}")

        if self.config.debug:
            logger.debug(
                "Test task schedule:" + json.dumps(self.test_task_schedule, indent="\t")
            )

        # Run the Training loop (which is defined in ContinualAssumption).
        results = self.main_loop(method)

        logger.info("Results summary:")
        logger.info(results.to_log_dict())
        logger.info(results.summary())
        method.receive_results(self, results=results)
        return results

        # Run the Test loop (which is defined in IncrementalAssumption).
        # results: RlResults = self.test_loop(method)

    def setup(self, stage: str = None) -> None:
        # Called before the start of each task during training, validation and
        # testing.
        super().setup(stage=stage)
        if stage in {"fit", None}:
            self.train_wrappers = self.create_train_wrappers()
            self.valid_wrappers = self.create_valid_wrappers()
        elif stage in {"test", None}:
            self.test_wrappers = self.create_test_wrappers()

    def prepare_data(self, *args, **kwargs) -> None:
        # We don't really download anything atm.
        if self.config is None:
            self.config = Config()
        super().prepare_data(*args, **kwargs)

    def train_dataloader(
        self, batch_size: int = None, num_workers: int = None
    ) -> ActiveEnvironment:
        """Create a training gym.Env/DataLoader for the current task.

        Parameters
        ----------
        batch_size : int, optional
            The batch size, which in this case is the number of environments to
            run in parallel. When `None`, the env won't be vectorized. Defaults
            to None.
        num_workers : int, optional
            The number of workers (processes) to use in the vectorized env. When
            None, the envs are run in sequence, which could be very slow. Only
            applies when `batch_size` is not None. Defaults to None.

        Returns
        -------
        GymDataLoader
            A (possibly vectorized) environment/dataloader for the current task.
        """
        if not self.has_prepared_data:
            self.prepare_data()
        # NOTE: We actually want to call setup every time, so we re-create the
        # wrappers for each task.
        # if not self.has_setup_fit:
        self.setup("fit")

        batch_size = batch_size or self.batch_size
        num_workers = num_workers if num_workers is not None else self.num_workers
        
        env_factory = partial(
            self._make_env,
            base_env=self.train_dataset,
            wrappers=self.train_wrappers,
            **self.base_env_kwargs,
        )
        env_dataloader = self._make_env_dataloader(
            env_factory,
            batch_size=batch_size,
            num_workers=num_workers,
            max_steps=self.steps_per_phase,
            max_episodes=self.train_max_episodes,
        )

        if self.monitor_training_performance:
            # NOTE: It doesn't always make sense to log stuff with the current task ID!
            wandb_prefix = "Train"
            if self.known_task_boundaries_at_train_time:
                wandb_prefix += f"/Task {self.current_task_id}"
            env_dataloader = MeasureRLPerformanceWrapper(
                env_dataloader, wandb_prefix=wandb_prefix
            )

        if self.config.render and batch_size is None:
            env_dataloader = RenderEnvWrapper(env_dataloader)

        self.train_env = env_dataloader
        # BUG: There is a mismatch between the train env's observation space and the
        # shape of its observations.
        # self.observation_space = self.train_env.observation_space

        return self.train_env

    def val_dataloader(
        self, batch_size: int = None, num_workers: int = None
    ) -> Environment:
        """Create a validation gym.Env/DataLoader for the current task.

        Parameters
        ----------
        batch_size : int, optional
            The batch size, which in this case is the number of environments to
            run in parallel. When `None`, the env won't be vectorized. Defaults
            to None.
        num_workers : int, optional
            The number of workers (processes) to use in the vectorized env. When
            None, the envs are run in sequence, which could be very slow. Only
            applies when `batch_size` is not None. Defaults to None.

        Returns
        -------
        GymDataLoader
            A (possibly vectorized) environment/dataloader for the current task.
        """
        if not self.has_prepared_data:
            self.prepare_data()
        self.setup("fit")

        env_factory = partial(
            self._make_env,
            base_env=self.val_dataset,
            wrappers=self.valid_wrappers,
            **self.base_env_kwargs,
        )
        env_dataloader = self._make_env_dataloader(
            env_factory,
            batch_size=batch_size or self.batch_size,
            num_workers=num_workers if num_workers is not None else self.num_workers,
            max_steps=self.steps_per_phase,
            # TODO: Create a new property to limit validation episodes?
            max_episodes=self.train_max_episodes,
        )

        if self.monitor_training_performance:
            # NOTE: We also add it here, just so it logs metrics to wandb.
            # NOTE: It doesn't always make sense to log stuff with the current task ID!
            wandb_prefix = "Valid"
            if self.known_task_boundaries_at_train_time:
                wandb_prefix += f"/Task {self.current_task_id}"
            env_dataloader = MeasureRLPerformanceWrapper(
                env_dataloader, wandb_prefix=wandb_prefix
            )

        self.val_env = env_dataloader
        return self.val_env

    def test_dataloader(
        self, batch_size: int = None, num_workers: int = None
    ) -> TestEnvironment:
        """Create the test 'dataloader/gym.Env' for all tasks.

        NOTE: This test environment isn't just for the current task, it actually
        contains the sequence of all tasks. This is different than the train or
        validation environments, since if the task labels are available at train
        time, then calling train/valid_dataloader` returns the envs for the
        current task only, and the `.fit` method is called once per task.

        This environment is also different in that it is wrapped with a Monitor,
        which we might eventually use to save the results/gifs/logs of the
        testing runs.

        Parameters
        ----------
        batch_size : int, optional
            The batch size, which in this case is the number of environments to
            run in parallel. When `None`, the env won't be vectorized. Defaults
            to None.
        num_workers : int, optional
            The number of workers (processes) to use in the vectorized env. When
            None, the envs are run in sequence, which could be very slow. Only
            applies when `batch_size` is not None. Defaults to None.

        Returns
        -------
        TestEnvironment
            A testing environment which keeps track of the performance of the
            actor and accumulates logs/statistics that are used to eventually
            create the 'Result' object.
        """
        if not self.has_prepared_data:
            self.prepare_data()
        self.setup("test")
        # BUG: gym.wrappers.Monitor doesn't want to play nice when applied to
        # Vectorized env, it seems..
        # FIXME: Remove this when the Monitor class works correctly with
        # batched environments.
        batch_size = batch_size or self.batch_size
        if batch_size is not None:
            logger.warning(
                UserWarning(
                    colorize(
                        f"WIP: Only support batch size of `None` (i.e., a single env) "
                        f"for the test environments of RL Settings at the moment, "
                        f"because the Monitor class from gym doesn't work with "
                        f"VectorEnvs. (batch size was {batch_size})",
                        "yellow",
                    )
                )
            )
            batch_size = None

        num_workers = num_workers if num_workers is not None else self.num_workers
        env_factory = partial(
            self._make_env,
            base_env=self.test_dataset,
            wrappers=self.test_wrappers,
            **self.base_env_kwargs,
        )
        # TODO: Pass the max_steps argument to this `_make_env_dataloader` method,
        # rather than to a `step_limit` on the TestEnvironment.
        env_dataloader = self._make_env_dataloader(
            env_factory, batch_size=batch_size, num_workers=num_workers,
        )
        if self.test_max_episodes is not None:
            raise NotImplementedError(f"TODO: Use `self.test_max_episodes`")

        test_loop_max_steps = self.test_max_steps // (batch_size or 1)
        # TODO: Find where to configure this 'test directory' for the outputs of
        # the Monitor.
        if wandb.run:
            test_dir = wandb.run.dir
        else:
            test_dir = "results"
        # TODO: Debug wandb Monitor integration.
        self.test_env = self.TestEnvironment(
            env_dataloader,
            task_schedule=self.test_task_schedule,
            directory=test_dir,
            step_limit=test_loop_max_steps,
            config=self.config,
            force=True,
            video_callable=None if wandb.run or self.config.render else False,
        )
        return self.test_env

    @property
    def phases(self) -> int:
        """The number of training 'phases', i.e. how many times `method.fit` will be
        called.

        In the case of ContinualRL and DiscreteTaskAgnosticRL, fit is only called once,
        with an environment that shifts between all the tasks. In IncrementalRL, fit is
        called once per task, while in TraditionalRL and MultiTaskRL, fit is called
        once.
        """
        return 1

    @property
    def steps_per_phase(self) -> Optional[int]:
        """Returns the number of steps per training "phase", i.e. the max number of
        (steps for now) that can be taken in the training environment passed to
        `Method.fit`

        In most settings, this is the same as `steps_per_task`.

        Returns
        -------
        Optional[int]
            `None` if `max_steps` is None, else `max_steps // phases`.
        """
        return (
            None
            if self.train_max_steps is None
            else self.train_max_steps // self.phases
        )

    @staticmethod
    def _make_env(
        base_env: Union[str, gym.Env, Callable[[], gym.Env]],
        wrappers: List[Callable[[gym.Env], gym.Env]] = None,
        **base_env_kwargs: Dict,
    ) -> gym.Env:
        """ Helper function to create a single (non-vectorized) environment. """
        env: gym.Env
        if isinstance(base_env, str):
            env = gym.make(base_env, **base_env_kwargs)
        elif isinstance(base_env, gym.Env):
            env = base_env
        elif callable(base_env):
            env = base_env(**base_env_kwargs)
        else:
            raise RuntimeError(
                f"base_env should either be a string, a callable, or a gym "
                f"env. (got {base_env})."
            )
        wrappers = wrappers or []
        for wrapper in wrappers:
            env = wrapper(env)
        return env

    def _make_env_dataloader(
        self,
        env_factory: Callable[[], gym.Env],
        batch_size: Optional[int],
        num_workers: Optional[int] = None,
        seed: Optional[int] = None,
        max_steps: Optional[int] = None,
        max_episodes: Optional[int] = None,
    ) -> GymDataLoader:
        """ Helper function for creating a (possibly vectorized) environment.

        """
        logger.debug(
            f"batch_size: {batch_size}, num_workers: {num_workers}, seed: {seed}"
        )

        env: Union[gym.Env, gym.vector.VectorEnv]
        if batch_size is None:
            env = env_factory()
        else:
            env = make_batched_env(
                env_factory,
                batch_size=batch_size,
                num_workers=num_workers,
                # TODO: Still debugging shared memory + custom spaces (e.g. Sparse).
                shared_memory=False,
            )
        if max_steps:
            env = ActionLimit(env, max_steps=max_steps)
        if max_episodes:
            env = EpisodeLimit(env, max_episodes=max_episodes)

        # Apply the "post-batch" wrappers:
        # from sequoia.common.gym_wrappers import ConvertToFromTensors
        # TODO: Only the BaselineMethod requires this, we should enable it only
        # from the BaselineMethod, and leave it 'off' by default.
        if self.add_done_to_observations:
            env = AddDoneToObservation(env)
        # # Convert the samples to tensors and move them to the right device.
        # env = ConvertToFromTensors(env)
        # env = ConvertToFromTensors(env, device=self.config.device)
        # Add a wrapper that converts numpy arrays / etc to Observations/Rewards
        # and from Actions objects to numpy arrays.
        env = TypedObjectsWrapper(
            env,
            observations_type=self.Observations,
            rewards_type=self.Rewards,
            actions_type=self.Actions,
        )
        # Create an IterableDataset from the env using the EnvDataset wrapper.
        dataset = EnvDataset(env)

        # Create a GymDataLoader for the EnvDataset.
        env_dataloader = GymDataLoader(dataset)

        if batch_size and seed:
            # Seed each environment with its own seed (based on the base seed).
            env.seed([seed + i for i in range(env_dataloader.num_envs)])
        else:
            env.seed(seed)

        return env_dataloader

    def create_train_wrappers(self) -> List[Callable[[gym.Env], gym.Env]]:
        """Get the list of wrappers to add to each training environment.

        The result of this method must be pickleable when using
        multiprocessing.

        Returns
        -------
        List[Callable[[gym.Env], gym.Env]]
            [description]
        """
        # We add a restriction to prevent users from getting data from
        # previous or future tasks.
        # TODO: This assumes that tasks all have the same length.
        return self._make_wrappers(
            base_env=self.train_dataset,
            task_schedule=self.train_task_schedule,
            # TODO: Removing this, but we have to check that it doesn't change when/how
            # the task boundaries are given to the Method.
            # sharp_task_boundaries=self.known_task_boundaries_at_train_time,
            task_labels_available=self.task_labels_at_train_time,
            transforms=self.transforms + self.train_transforms,
            starting_step=0,
            max_steps=self.train_max_steps,
            new_random_task_on_reset=self.stationary_context,
        )

    def create_valid_wrappers(self) -> List[Callable[[gym.Env], gym.Env]]:
        """Get the list of wrappers to add to each validation environment.

        The result of this method must be pickleable when using
        multiprocessing.

        Returns
        -------
        List[Callable[[gym.Env], gym.Env]]
            [description]

        TODO: Decide how this 'validation' environment should behave in
        comparison with the train and test environments.
        """
        return self._make_wrappers(
            base_env=self.val_dataset,
            task_schedule=self.val_task_schedule,
            # sharp_task_boundaries=self.known_task_boundaries_at_train_time,
            task_labels_available=self.task_labels_at_train_time,
            transforms=self.transforms + self.val_transforms,
            starting_step=0,
            # TODO: Should there be a limit on the validation steps/episodes?
            max_steps=self.train_max_steps,
            new_random_task_on_reset=self.stationary_context,
        )

    def create_test_wrappers(self) -> List[Callable[[gym.Env], gym.Env]]:
        """Get the list of wrappers to add to a single test environment.

        The result of this method must be pickleable when using
        multiprocessing.

        Returns
        -------
        List[Callable[[gym.Env], gym.Env]]
            [description]
        """
        return self._make_wrappers(
            base_env=self.test_dataset,
            task_schedule=self.test_task_schedule,
            # sharp_task_boundaries=self.known_task_boundaries_at_test_time,
            task_labels_available=self.task_labels_at_test_time,
            transforms=self.transforms + self.test_transforms,
            starting_step=0,
            max_steps=self.test_max_steps,
            new_random_task_on_reset=self.stationary_context,
        )

    def _make_wrappers(
        self,
        base_env: Union[str, gym.Env, Callable[[], gym.Env]],
        task_schedule: Dict[int, Dict],
        # sharp_task_boundaries: bool,
        task_labels_available: bool,
        transforms: List[Transforms] = None,
        starting_step: int = None,
        max_steps: int = None,
        new_random_task_on_reset: bool = False,
    ) -> List[Callable[[gym.Env], gym.Env]]:
        """ helper function for creating the train/valid/test wrappers.

        These wrappers get applied *before* the batching, if applicable.
        """
        wrappers: List[Callable[[gym.Env], gym.Env]] = []

        # TODO: Add some kind of Wrapper around the dataset to make it
        # semi-supervised?

        if self.max_episode_steps:
            wrappers.append(
                partial(TimeLimit, max_episode_steps=self.max_episode_steps)
            )

        # NOTE: Removing this 'ActionLimit' from the 'pre-batch' wrappers.
        # wrappers.append(partial(ActionLimit, max_steps=max_steps))

        # if is_classic_control_env(base_env):
        # If we are in a classic control env, and we dont want the state to
        # be fully-observable (i.e. we want pixel observations rather than
        # getting the pole angle, velocity, etc.), then add the
        # PixelObservation wrapper to the list of wrappers.
        # if self.force_pixel_observations:
        #     wrappers.append(PixelObservationWrapper)

        if is_atari_env(base_env):
            # TODO: Test & Debug this: Adding the Atari preprocessing wrapper.
            # TODO: Figure out the differences (if there are any) between the
            # AtariWrapper from SB3 and the AtariPreprocessing wrapper from gym.
            wrappers.append(AtariWrapper)
            # wrappers.append(AtariPreprocessing)

            # # TODO: Not sure if we should add the transforms to the env here!
            # # BUG: In the case where `train_envs` is passed (to the IncrementalRL
            # # setting), and contains functools.partial for some env, then we have a
            # # problem because we can't tell if we need to add some wrapper like
            # # PixelObservations!
            # assert False, (
            #     f"Can't tell if we should be adding a PixelObservationsWrapper if "
            #     f"the env isn't somethign we know how to handle!: {self.dataset}"
            # )

        if transforms:
            # Apply image transforms if the env will have image-like obs space
            # Wrapper to 'wrap' the observation space into an Image space (subclass of
            # Box with useful fields like `c`, `h`, `w`, etc.)
            wrappers.append(ImageObservations)
            # Wrapper to apply the image transforms to the env.
            wrappers.append(partial(TransformObservation, f=transforms))

        # TODO: BUG: Currently still need to add a CL wrapper (so that we can then
        # create the task schedule) even when `task_schedule` here is empty! (This is
        # because this is called in `__post_init__()`, where `train_task_schedule is
        # still empty.`)
        if task_schedule is not None:
            # Add a wrapper which will add non-stationarity to the environment.
            # The "task" transitions will either be sharp or smooth.
            # In either case, the task ids for each sample are added to the
            # observations, and the dicts containing the task information (e.g. the
            # current values of the env attributes from the task schedule) get added
            # to the 'info' dicts.
            if self.smooth_task_boundaries:
                # Add a wrapper that creates smooth tasks.
                cl_wrapper = SmoothTransitions
            else:
                assert self.nb_tasks >= 1
                # Add a wrapper that creates sharp tasks.
                cl_wrapper = MultiTaskEnvironment

            assert starting_step is not None
            assert max_steps is not None

            wrappers.append(
                partial(
                    cl_wrapper,
                    noise_std=self.task_noise_std,
                    task_schedule=task_schedule,
                    add_task_id_to_obs=True,
                    add_task_dict_to_info=True,
                    starting_step=starting_step,
                    new_random_task_on_reset=new_random_task_on_reset,
                    max_steps=max_steps,
                )
            )
            # If the task labels aren't available, we then add another wrapper that
            # hides that information (setting both of them to None) and also marks
            # those spaces as `Sparse`.
            if not task_labels_available:
                # NOTE: This sets the task labels to None, rather than removing
                # them entirely.
                # wrappers.append(RemoveTaskLabelsWrapper)
                wrappers.append(HideTaskLabelsWrapper)

        return wrappers

    def _get_objective_scaling_factor(self) -> float:
        """ Return the factor to be multiplied with the mean reward per episode
        in order to produce a 'performance score' between 0 and 1.

        Returns
        -------
        float
            The scaling factor to use.
        """
        # TODO: remove this, currently used just so we can get a 'scaling factor' to use
        # to scale the 'mean reward per episode' to a score between 0 and 1.
        # TODO: Add other environments, for instance 1/200 for cartpole.
        # TODO: Rework this so its based on the reward threshold!
        max_reward_per_episode = 1
        if isinstance(self.dataset, str) and self.dataset.startswith("MetaMonsterKong"):
            max_reward_per_episode = 100
        elif isinstance(self.dataset, str) and self.dataset == "CartPole-v0":
            max_reward_per_episode = 200
        else:
            warnings.warn(
                RuntimeWarning(
                    f"Unable to determine the right scaling factor to use for dataset "
                    f"{self.dataset} when calculating the performance score! "
                    f"The CL Score of this run will most probably not be accurate."
                )
            )
        return 1 / max_reward_per_episode

    def _get_simple_name(self, env_name_or_id: str) -> Optional[str]:
        """ Returns the 'simple name' for the given environment ID.
        For example, when passed "CartPole-v0", returns "cartpole".

        When not found, returns None.
        """
        if env_name_or_id in self.available_datasets.keys():
            return env_name_or_id

        if env_name_or_id in self.available_datasets.values():
            simple_name: str = [
                k for k, v in self.available_datasets.items() if v == env_name_or_id
            ][0]
            return simple_name
        return None


def _load_task_schedule(file_path: Path) -> Dict[int, Dict]:
    """ Load a task schedule from the given path. """
    with open(file_path) as f:
        task_schedule = json.load(f)
        return {int(k): task_schedule[k] for k in sorted(task_schedule.keys())}


if __name__ == "__main__":
    ContinualRLSetting.main()


def find_matching_dataset(
    available_datasets: Dict[str, Union[str, Any]], dataset: str
) -> Optional[Union[str, Any]]:
    """ Compares `dataset` with the keys in the `available_datasets` dict and return the
    value of the matching key if found, else returns None.
    """
    if dataset in available_datasets:
        return available_datasets[dataset]

    if not isinstance(dataset, str):
        raise NotImplementedError(dataset)

    chosen_env_name, _, chosen_version = dataset.partition("-v")
    for key, env_id in available_datasets.items():
        if dataset == key:
            assert False, "this should be reached, since we do that check above"

        env_name, _, env_version = key.partition("-v")
        if chosen_version:
            # chosen: half_cheetah
            # key: HalfCheetah-v2
            # HalfCheetah-v2
            # halfcheetah-v2
            # half_cheetah_v2
            if chosen_version != env_version:
                continue
            if names_match(chosen_env_name, env_name):
                return env_id
        elif names_match(chosen_env_name, env_name):
            # Look for matching entries with that name, and select the highest
            # available version.
            datasets_with_that_name = {
                other_key: other_env_id
                for other_key, other_env_id in available_datasets.items()
                if names_match(chosen_env_name, other_key.partition("-v")[0])
            }
            if len(datasets_with_that_name) == 1:
                return env_id
            versions = {
                other_key: int(other_key.partition("-v")[-1])
                for other_key in datasets_with_that_name
            }
            return max(datasets_with_that_name, key=versions.get)

    closest_matches = difflib.get_close_matches(dataset, available_datasets)
    if closest_matches:
        closest_match_key: str = closest_matches[0]
        closest_match: Union[str, Any] = available_datasets[closest_match_key]
        if chosen_version:
            # Find the 'version' number of the closest match, and check that it fits.
            closest_match_version = closest_match_key.partition("-v")[-1]
            if not closest_match_version:
                assert isinstance(closest_match, str)
                closest_match_version = closest_match.partition("-v")[-1]

            if chosen_version == closest_match_version:
                return closest_match

            raise gym.error.UnregisteredEnv(
                f"Can't find any matching entries for chosen dataset {dataset} "
                f"with that same version (closest entries: {closest_matches}) "
            )

        warnings.warn(
            RuntimeWarning(
                f"Can't find matching entry for chosen dataset {dataset}, using "
                f"closest match: {closest_match}"
            )
        )
        return closest_match
        # raise RuntimeError(f"Can't find any matching entries for chosen dataset {dataset}. "
        #                 f"Closest entries: {closest_matches}")

    raise gym.error.UnregisteredEnv(
        f"Can't find any matching entries for chosen dataset {dataset}."
    )
    # assert False, (dataset, closest_matches)

