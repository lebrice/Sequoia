import itertools
import json
import math
import warnings
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, List, Optional, Sequence, Type, Union

import gym
from gym import spaces
from gym.utils import colorize
from gym.wrappers import TimeLimit
from sequoia.common import Config
from sequoia.common.gym_wrappers import (
    AddDoneToObservation,
    MultiTaskEnvironment,
    SmoothTransitions,
    TransformObservation,
)
from sequoia.common.gym_wrappers.batch_env.tile_images import tile_images
from sequoia.common.gym_wrappers.env_dataset import EnvDataset
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
from sequoia.common.spaces import Sparse
from sequoia.common.spaces.named_tuple import NamedTupleSpace
from sequoia.common.transforms import Transforms
from sequoia.settings.rl import RLSetting
from sequoia.settings.assumptions.incremental import (
    IncrementalSetting,
    TaskResults,
    TaskSequenceResults,
    TestEnvironment,
)
from sequoia.settings.base import Method
from sequoia.utils import get_logger
from simple_parsing import choice, field, list_field
from simple_parsing.helpers import dict_field
from stable_baselines3.common.atari_wrappers import AtariWrapper

from sequoia.settings.rl import ActiveEnvironment
from .environment import GymDataLoader
from .make_env import make_batched_env
from .results import RLResults
from .wrappers import HideTaskLabelsWrapper, TypedObjectsWrapper

logger = get_logger(__file__)

# TODO: Implement a get_metrics (ish) in the Environment, not on the Setting!
# TODO: The validation environment will also call the on_task_switch when it
# reaches a task boundary, and there isn't currently a way to distinguish if
# that method is being called because of the training or because of the
# validation environment.

from gym.envs.classic_control import CartPoleEnv, PendulumEnv, MountainCarEnv

from .tasks import make_task_for_env

# Type alias for the Environment returned by `train/val/test_dataloader`.
Environment = ActiveEnvironment[
    "ContinualRLSetting.Observations",
    "ContinualRLSetting.Observations",
    "ContinualRLSetting.Rewards",
]

# TODO: Update the 'available environments' to show all available gym envs? or only
# those that are explicitly supported ?
env_ids = [env_spec.id for env_spec in gym.envs.registry.all()]
available_environments = env_ids

# TODO: Fix this, shouldn't inherit from IncrementalSetting.
@dataclass
class ContinualRLSetting(RLSetting, IncrementalSetting):
    """ Reinforcement Learning Setting where the environment changes over time.

    This is an Active setting which uses gym environments as sources of data.
    These environments' attributes could change over time following a task
    schedule. An example of this could be that the gravity increases over time
    in cartpole, making the task progressively harder as the agent interacts with
    the environment.
    """

    # The type of results returned by an RL experiment.
    Results: ClassVar[Type[RLResults]] = RLResults

    @dataclass(frozen=True)
    class Observations(IncrementalSetting.Observations):
        """ Observations in a continual RL Setting. """

        # Just as a reminder, these are the fields defined in the base classes:
        # x: Tensor
        # task_labels: Union[Optional[Tensor], Sequence[Optional[Tensor]]] = None

        # The 'done' part of the 'step' method. We add this here in case a
        # method were to iterate on the environments in the dataloader-style so
        # they also have access to those (i.e. for the BaselineMethod).
        done: Optional[Sequence[bool]] = None
        # Same, for the 'info' portion of the result of 'step'.
        # TODO: If we add the 'task space' (with all the attributes, for instance
        # then add it to the observations using the `AddInfoToObservations`.
        # info: Optional[Sequence[Dict]] = None

    # Image transforms to use.
    transforms: List[Transforms] = list_field()

    # Class variable that holds the dict of available environments.
    available_datasets: ClassVar[Dict[str, str]] = {
        "cartpole": "CartPole-v0",
        "pendulum": "Pendulum-v0",
        "breakout": "Breakout-v0",
        # "duckietown": "Duckietown-straight_road-v0"
        # NOTE: We register both the 'true' environment ID, as well as a simpler name:
        "half_cheetah": "ContinualHalfCheetah-v2",
        "HalfCheetah-v2": "ContinualHalfCheetah-v2",
        "hopper": "ContinualHopper-v2",
        "Hopper-v2": "ContinualHopper-v2",
        "walker2d": "ContinualWalker2d-v2",
        "Walker2d-v2": "ContinualWalker2d-v2",
    }
    # TODO: Add breakout to 'available_datasets' only when atari_py is installed.

    # Which environment (a.k.a. "dataset") to learn on.
    # The dataset could be either a string (env id or a key from the
    # available_datasets dict), a gym.Env, or a callable that returns a
    # single environment.
    dataset: str = "CartPole-v0"

    # Environment/dataset to use for validation. Defaults to the same as `dataset`.
    val_dataset: Optional[str] = None
    # Environment/dataset to use for testing. Defaults to the same as `dataset`.
    test_dataset: Optional[str] = None

    # The number of "tasks", which in the case of Continual settings means the number of
    # base tasks that are interpolated in order to create the training, valid and test
    # nonstationarities.
    # By default 1 for this setting, meaning that the context is a linear interpolation
    # between the start context (usually the default task for the environment) and a
    # randomly sampled task.
    nb_tasks: int = field(1, alias=["n_tasks", "num_tasks"])

    # Max number of steps per task. (Also acts as the "length" of the training
    # and validation "Datasets")
    max_steps: int = 100_000
    # Maximum episodes per task.
    # TODO: Test that the limit on the number of episodes actually works.
    max_episodes: Optional[int] = None
    # Number of steps per task. When left unset and when `max_steps` is set,
    # takes the value of `max_steps` divided by `nb_tasks`.
    steps_per_task: Optional[int] = None
    # (WIP): Number of episodes per task.
    episodes_per_task: Optional[int] = None

    # Total number of steps in the test loop. (Also acts as the "length" of the testing
    # environment.)
    test_steps: int = 10_000
    # Number of steps per task in the test loop. When left unset and when `test_steps`
    # is set, takes the value of `test_steps` divided by `nb_tasks`.
    test_steps_per_task: Optional[int] = None

    # Standard deviation of the multiplicative Gaussian noise that is used to
    # create the values of the env attributes for each task.
    task_noise_std: float = 0.2

    # Wether the task boundaries are smooth or sudden.
    smooth_task_boundaries: bool = True

    # NOTE: THIS ARG IS DEPRECATED! Only keeping it here so previous config yaml files
    # don't cause a crash.
    observe_state_directly: Optional[bool] = None

    # When True, will attempt to extract pixel observations from the environment,
    # wither by adding a PixelObservation (for classic control envs for instance) or
    # by passing the right argument to the env constructor when possible.
    force_pixel_observations: bool = False
    """ Wether to use the "pixel" version of `self.dataset`.
    When `False`, does nothing.
    When `True`, will do one of the following, depending on the choice of environment:
    - For classic control envs, it adds a `PixelObservationsWrapper` to the env.
    - For atari envs:
        - If `self.dataset` is a regular atari env (e.g. "Breakout-v0"), does nothing.
        - if `self.dataset` is the 'RAM' version of an atari env, raises an error.
    - For mujoco envs, this raises a NotImplementedError, as we don't yet know how to
      make a pixel-version the Mujoco Envs.
    - For other envs:
        - If the environment's observation space appears to be image-based, an error
          will be raised.
        - If the environment's observation space doesn't seem to be image-based, does
          nothing.
    """

    force_state_observations: bool = False
    """ Wether to use the "state" version of `self.dataset`.
    When `False`, does nothing.
    When `True`, will do one of the following, depending on the choice of environment:
    - For classic control envs, it does nothing, as they are already state-based.
    - TODO: For atari envs, the 'RAM' version of the chosen env will be used.
    - For mujoco envs, it doesn nothing, as they are already state-based.
    - For other envs, if this is set to True, then
        - If the environment's observation space appears to be image-based, an error
          will be raised.
        - If the environment's observation space doesn't seem to be image-based, does
          nothing.
    """

    # Path to a json file from which to read the train task schedule.
    train_task_schedule_path: Optional[Path] = None
    # Path to a json file from which to read the validation task schedule.
    valid_task_schedule_path: Optional[Path] = None
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
    transforms: Optional[List[Transforms]] = None

    # Transforms to be applied to the training datasets.
    train_transforms: Optional[List[Transforms]] = None
    # Transforms to be applied to the validation datasets.
    val_transforms: Optional[List[Transforms]] = None
    # Transforms to be applied to the testing datasets.
    test_transforms: Optional[List[Transforms]] = None

    # NOTE: Added this `cmd=False` option to mark that we don't want to generate
    # any command-line arguments for these fields.
    train_task_schedule: Dict[int, Dict[str, float]] = dict_field(cmd=False)
    valid_task_schedule: Dict[int, Dict[str, float]] = dict_field(cmd=False)
    test_task_schedule: Dict[int, Dict[str, float]] = dict_field(cmd=False)

    # TODO: Naming is a bit inconsistent, using `valid` here, whereas we use `val`
    # elsewhere.
    train_wrappers: List[Callable[[gym.Env], gym.Env]] = list_field(cmd=False)
    valid_wrappers: List[Callable[[gym.Env], gym.Env]] = list_field(cmd=False)
    test_wrappers: List[Callable[[gym.Env], gym.Env]] = list_field(cmd=False)

    # keyword arguments to be passed to the base environment through gym.make(base_env, **kwargs).
    base_env_kwargs: Dict = dict_field(cmd=False)

    batch_size: Optional[int] = field(default=None, cmd=False)
    num_workers: Optional[int] = field(default=None, cmd=False)

    # The function to use to sample tasks for the given environment.
    task_sampling_function: Callable[[gym.Env, int, List[int]], Any] = field(
        None, cmd=False
    )

    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)
        self._new_random_task_on_reset: bool = False

        if self.observe_state_directly is not None:
            warnings.warn(DeprecationWarning(
                "The `observe_state_directly` field is deprecated! Use "
                "`force_state_observations` or `force_pixel_observations` instead."
            ))
            # For now, set the equivalent values for the envs we know how to handle.
            if is_classic_control_env(self.dataset) and not self.observe_state_directly:
                self.force_pixel_observations = True
            elif is_monsterkong_env(self.dataset):
                if self.observe_state_directly:
                    self.force_state_observations = True
                else:
                    self.force_pixel_observations = True

        if self.force_pixel_observations and self.force_state_observations:
            raise RuntimeError(
                "Can't set both `force_pixel_observations` and "
                "`force_state_observations`!"
            )

        # Post processing of the 'dataset' field:
        if self.dataset in self.available_datasets.keys():
            # the environment name was passed, rather than an id
            # (e.g. 'cartpole' -> 'CartPole-v0").
            self.dataset = self.available_datasets[self.dataset]

        elif self.dataset in self.available_datasets.values():
            # A known environment class or some callable was passed, this is fine.
            pass

        # elif if isinstance(self.dataset, str) and camel_case(self.dataset) in self.available_datasets.keys()
        else:
            # The passed dataset is assumed to be an environment ID, but it
            # wasn't in the dict of available datasets! We issue a warning, but
            # proceed to let the user use whatever environment they want to.
            logger.warning(
                UserWarning(
                    f"The chosen dataset/environment ({self.dataset}) isn't in the "
                    f"available_datasets dict, so we can't garantee this will work!"
                )
            )

        if isinstance(self.dataset, gym.Env) and self.batch_size:
            raise RuntimeError(
                "Batch size should be None when a gym.Env "
                "object is passed as `dataset`."
            )
        if not isinstance(self.dataset, (str, gym.Env)) and not callable(self.dataset):
            raise RuntimeError(
                f"`dataset` must be either a string, a gym.Env, or a callable. "
                f"(got {self.dataset})"
            )

        self.val_dataset = self.val_dataset or self.dataset
        self.test_dataset = self.test_dataset or self.dataset

        # Set the number of tasks depending on the increment, and vice-versa.
        # (as only one of the two should be used).
        assert self.max_steps, "assuming this should always be set, for now."
        # TODO: Clean this up, not super clear what options take precedence on
        # which other options.

        # Load the task schedules from the corresponding files, if present.
        if self.train_task_schedule_path:
            self.train_task_schedule = self.load_task_schedule(
                self.train_task_schedule_path
            )

        if self.valid_task_schedule_path:
            self.valid_task_schedule = self.load_task_schedule(
                self.valid_task_schedule_path
            )

        if self.test_task_schedule_path:
            self.test_task_schedule = self.load_task_schedule(
                self.test_task_schedule_path
            )

        if self.train_task_schedule:
            if self.steps_per_task is not None:
                # If steps per task was passed, then we overwrite the keys of the tasks
                # schedule.
                self.train_task_schedule = {
                    i * self.steps_per_task: self.train_task_schedule[step]
                    for i, step in enumerate(sorted(self.train_task_schedule.keys()))
                }
            else:
                # A task schedule was passed: infer the number of tasks from it.
                change_steps = sorted(self.train_task_schedule.keys())
                assert 0 in change_steps, "Schedule needs a task at step 0."
                # TODO: @lebrice: I guess we have to assume that the interval
                # between steps is constant for now? Do we actually depend on this
                # being the case? I think steps_per_task is only really ever used
                # for creating the task schedule, which we already have in this
                # case.
                assert (
                    len(change_steps) >= 2
                ), "WIP: need a minimum of two tasks in the task schedule for now."
                self.steps_per_task = change_steps[1] - change_steps[0]
                # Double-check that this is the case.
                for i in range(len(change_steps) - 1):
                    if change_steps[i + 1] - change_steps[i] != self.steps_per_task:
                        raise NotImplementedError(
                            "WIP: This might not work yet if the tasks aren't "
                            "equally spaced out at a fixed interval."
                        )

            nb_tasks = len(self.train_task_schedule)
            if self.smooth_task_boundaries:
                # NOTE: When in a ContinualRLSetting with smooth task boundaries,
                # the last entry in the schedule represents the state of the env at
                # the end of the "task". When there are clear task boundaries (i.e.
                # when in 'Class'/Task-Incremental RL), the last entry is the start
                # of the last task.
                nb_tasks -= 1
            if self.nb_tasks not in {0, 1}:
                if self.nb_tasks != nb_tasks:
                    raise RuntimeError(
                        f"Passed number of tasks {self.nb_tasks} doesn't match the "
                        f"number of tasks deduced from the task schedule ({nb_tasks})"
                    )
            self.nb_tasks = nb_tasks

            self.max_steps = max(self.train_task_schedule.keys())
            if not self.smooth_task_boundaries:
                # See above note about the last entry.
                self.max_steps += self.steps_per_task

        elif self.nb_tasks:
            if self.steps_per_task:
                self.max_steps = self.nb_tasks * self.steps_per_task
            elif self.max_steps:
                self.steps_per_task = self.max_steps // self.nb_tasks

        elif self.steps_per_task:
            if self.nb_tasks:
                self.max_steps = self.nb_tasks * self.steps_per_task
            elif self.max_steps:
                self.nb_tasks = self.max_steps // self.steps_per_task

        elif self.max_steps:
            if self.nb_tasks:
                self.steps_per_task = self.max_steps // self.nb_tasks
            elif self.steps_per_task:
                self.nb_tasks = self.max_steps // self.steps_per_task
            else:
                self.nb_tasks = 1
                self.steps_per_task = self.max_steps

        if not all([self.nb_tasks, self.max_steps, self.steps_per_task]):
            raise RuntimeError(
                "You need to provide at least two of 'max_steps', "
                "'nb_tasks', or 'steps_per_task'."
            )

        assert self.nb_tasks == self.max_steps // self.steps_per_task

        if self.test_task_schedule:
            if 0 not in self.test_task_schedule:
                raise RuntimeError("Task schedules needs to include an initial task.")

            if self.test_steps_per_task is not None:
                # If steps per task was passed, then we overwrite the number of steps
                # for each task in the schedule to match.
                self.test_task_schedule = {
                    i * self.test_steps_per_task: self.test_task_schedule[step]
                    for i, step in enumerate(sorted(self.test_task_schedule.keys()))
                }

            change_steps = sorted(self.test_task_schedule.keys())
            assert 0 in change_steps, "Schedule needs to include task at step 0."

            nb_test_tasks = len(change_steps)
            if self.smooth_task_boundaries:
                nb_test_tasks -= 1
            assert (
                nb_test_tasks == self.nb_tasks
            ), "nb of tasks should be the same for train and test."

            self.test_steps_per_task = change_steps[1] - change_steps[0]
            for i in range(self.nb_tasks - 1):
                if change_steps[i + 1] - change_steps[i] != self.test_steps_per_task:
                    raise NotImplementedError(
                        "WIP: This might not work yet if the test tasks aren't "
                        "equally spaced out at a fixed interval."
                    )

            self.test_steps = max(change_steps)
            if not self.smooth_task_boundaries:
                # See above note about the last entry.
                self.test_steps += self.test_steps_per_task

        elif self.test_steps_per_task is None:
            # This is basically never the case, since the test_steps defaults to 10_000.
            assert (
                self.test_steps
            ), "need to set one of test_steps or test_steps_per_task"
            self.test_steps_per_task = self.test_steps // self.nb_tasks
        else:
            # FIXME: This is too complicated for what is is.
            # Check that the test steps must either be the default value, or the right
            # value to use in this case.
            assert self.test_steps in {10_000, self.test_steps_per_task * self.nb_tasks}
            assert (
                self.test_steps_per_task
            ), "need to set one of test_steps or test_steps_per_task"
            self.test_steps = self.test_steps_per_task * self.nb_tasks

        assert self.test_steps // self.test_steps_per_task == self.nb_tasks

        if self.smooth_task_boundaries:
            # If we're operating in the 'Online/smooth task transitions' "regime",
            # then there is only one "task", and we don't have task labels.
            # TODO: HOWEVER, the task schedule could/should be able to have more
            # than one non-stationarity! This indicates a need for a distinction
            # between 'tasks' and 'non-stationarities' (changes in the env).
            self.known_task_boundaries_at_train_time = False
            self.known_task_boundaries_at_test_time = False
            self.task_labels_at_train_time = False
            self.task_labels_at_test_time = False
            # self.steps_per_task = self.max_steps

        # Task schedules for training / validation and testing.

        # Create a temporary environment so we can extract the spaces and create
        # the task schedules.
        with self._make_env(
            self.dataset, self._temp_wrappers(), **self.base_env_kwargs
        ) as temp_env:
            self._setup_fields_using_temp_env(temp_env)
        del temp_env

        self.train_env: gym.Env
        self.valid_env: gym.Env
        self.test_env: gym.Env

    def _setup_fields_using_temp_env(self, temp_env: MultiTaskEnvironment):
        """ Setup some of the fields on the Setting using a temporary environment.

        This temporary environment only lives during the __post_init__() call.
        """
        # For now since we always have a CL-type wrapper (either MultiTaskEnv (or
        # AddTaskIdWrapper in IncrementalRL, the observation space should already be a
        # NamedTupleSpace.
        # NOTE: We use 'NamedTupleSpace', rather than dict, because we need Observations
        # to be objects (in this case, Batch objects, which are basically dataclasses
        # with tensor fields). This is because we need inheritance to work for the
        # objects given by the envs, so that methods can accept new subclasses too.
        assert isinstance(temp_env.observation_space, NamedTupleSpace)

        # FIXME: Replacing the observation space dtypes from their original
        # 'generated' NamedTuples to self.Observations. The alternative
        # would be to add another argument to the MultiTaskEnv wrapper, to
        # pass down a dtype to be set on its observation_space's `dtype`
        # attribute, which would be ugly.
        temp_env.observation_space.dtype = self.Observations

        # Populate the task schedules created above.
        if not self.train_task_schedule:
            train_change_steps = [i * self.steps_per_task for i in range(self.nb_tasks)]
            assert len(train_change_steps) == self.nb_tasks

            if self.smooth_task_boundaries:
                # Add a last 'task' at the end of the 'epoch', so that the
                # env changes smoothly right until the end.
                train_change_steps.append(self.max_steps)

            self.train_task_schedule = self.create_task_schedule(
                temp_env=temp_env, change_steps=train_change_steps
            )

        assert self.train_task_schedule is not None
        # The validation task schedule is the same as the one used in
        # training by default.
        if not self.valid_task_schedule:
            self.valid_task_schedule = deepcopy(self.train_task_schedule)

        if not self.test_task_schedule:
            # The test task schedule is by default the same as in validation
            # except that the interval between the tasks may be different,
            # depending on the value of `self.test_steps_per_task`.
            valid_steps = sorted(self.valid_task_schedule.keys())
            valid_tasks = [self.valid_task_schedule[step] for step in valid_steps]
            self.test_task_schedule = {
                i * self.test_steps_per_task: deepcopy(task)
                for i, task in enumerate(valid_tasks)
            }

        space_dict = dict(temp_env.observation_space.items())
        task_label_space = spaces.Discrete(self.nb_tasks)
        if not self.task_labels_at_train_time or not self.task_labels_at_test_time:
            task_label_space = Sparse(task_label_space)
        space_dict["task_labels"] = task_label_space

        # FIXME: Temporarily, we will actually set the task label space, since there
        # appears to be an error when using monsterkong space.
        observation_space = NamedTupleSpace(spaces=space_dict, dtype=self.Observations)
        self.observation_space = observation_space
        # Set the spaces using the temp env.
        # self.observation_space = temp_env.observation_space
        self.action_space = temp_env.action_space
        self.reward_range = temp_env.reward_range
        self.reward_space = getattr(
            temp_env,
            "reward_space",
            spaces.Box(low=self.reward_range[0], high=self.reward_range[1], shape=()),
        )

    def create_task_schedule(
        self, temp_env: gym.Env, change_steps: List[int]
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
            task = self.sample_task(env=temp_env, step=step, change_steps=change_steps)
            task_schedule[step] = task

        return task_schedule

    def sample_task(
        self, env: gym.Env, step: int, change_steps: List[int]
    ) -> Dict[str, Any]:
        if self.task_sampling_function:
            return self.task_sampling_function(
                env,
                step=step,
                change_steps=change_steps,
                seed=self.config.seed if self.config else None,
            )

        # Check if there is a registered handler for this type of environment.
        # TODO: Maybe use the type of the env, rather then env.unwrapped, and have the
        # 'base' callable defer the call to the wrapped env?
        if make_task_for_env.dispatch(type(env.unwrapped)) is make_task_for_env.dispatch(object):
            warnings.warn(
                RuntimeWarning(
                    f"Don't yet know how to create a task for env {env}, will use the environment as-is."
                )
            )

        # Use this singledispatch function to select how to sample a task depending on
        # the type of environment.
        return make_task_for_env(
            env.unwrapped,
            step=step,
            change_steps=change_steps,
            seed=self.config.seed if self.config else None,
        )

    def apply(
        self, method: Method, config: Config = None
    ) -> "ContinualRLSetting.Results":
        """Apply the given method on this setting to producing some results. """
        # Use the supplied config, or parse one from the arguments that were
        # used to create `self`.
        self.config: Config
        if config is not None:
            self.config = config
            logger.debug(f"Using Config {self.config}")
        elif isinstance(getattr(method, "config", None), Config):
            self.config = method.config
            logger.debug(f"Using Config from the Method: {self.config}")
        else:
            logger.debug("Parsing the Config from the command-line.")
            self.config = Config.from_args(self._argv, strict=False)
            logger.debug(f"Resulting Config: {self.config}")

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
        if self._new_random_task_on_reset:
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

        # Run the Training loop (which is defined in IncrementalSetting).
        results = self.main_loop(method)

        logger.info("Results summary:")
        logger.info(results.to_log_dict())
        logger.info(results.summary())
        method.receive_results(self, results=results)
        return results

        # Run the Test loop (which is defined in IncrementalSetting).
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
            base_env=self.dataset,
            wrappers=self.train_wrappers,
            **self.base_env_kwargs,
        )
        env_dataloader = self._make_env_dataloader(
            env_factory,
            batch_size=batch_size,
            num_workers=num_workers,
            max_steps=self.steps_per_phase,
            max_episodes=self.episodes_per_task,
        )

        if self.monitor_training_performance:
            from sequoia.settings.sl.class_incremental.measure_performance_wrapper import (
                MeasureRLPerformanceWrapper,
            )

            env_dataloader = MeasureRLPerformanceWrapper(
                env_dataloader, wandb_prefix=f"Train/Task {self.current_task_id}"
            )

        self.train_env = env_dataloader
        # BUG: There is a mismatch between the train env's observation space and the
        # shape of its observations.
        self.observation_space = self.train_env.observation_space

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
            max_episodes=self.episodes_per_task,
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
            logger.warn(
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
        # TODO: We should probably change the max_steps depending on the
        # batch size of the env.
        test_loop_max_steps = self.test_steps // (batch_size or 1)
        # TODO: Find where to configure this 'test directory' for the outputs of
        # the Monitor.
        test_dir = "results"
        # TODO: Debug wandb Monitor integration.
        self.test_env = ContinualRLTestEnvironment(
            env_dataloader,
            task_schedule=self.test_task_schedule,
            directory=test_dir,
            step_limit=test_loop_max_steps,
            config=self.config,
            force=True,
            video_callable=None if self.config.render else False,
        )
        return self.test_env

    @property
    def phases(self) -> int:
        """The number of training 'phases', i.e. how many times `method.fit` will be
        called.

        In the case of ContinualRL, fit is only called once, with an environment that
        shifts between all the tasks.
        """
        return 1

    @property
    def steps_per_phase(self) -> Optional[int]:
        """Returns the number of steps per training "phase".

        In most settings, this is the same as `steps_per_task`.

        Returns
        -------
        Optional[int]
            `None` if `max_steps` is None, else `max_steps // phases`.
        """
        # TODO: This doesn't work when the schedule has tasks of different length.
        return None if self.max_steps is None else self.max_steps // self.phases

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
            from sequoia.common.gym_wrappers.action_limit import ActionLimit

            env = ActionLimit(env, max_steps=max_steps)
        if max_episodes:
            from sequoia.common.gym_wrappers.episode_limit import EpisodeLimit

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
        dataset = EnvDataset(env, max_steps=max_steps, max_episodes=max_episodes,)

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
        starting_step = self.current_task_id * self.steps_per_task
        max_steps = starting_step + self.steps_per_task - 1
        return self._make_wrappers(
            base_env=self.dataset,
            task_schedule=self.train_task_schedule,
            # TODO: Removing this, but we have to check that it doesn't change when/how
            # the task boundaries are given to the Method.
            # sharp_task_boundaries=self.known_task_boundaries_at_train_time,
            task_labels_available=self.task_labels_at_train_time,
            transforms=self.train_transforms,
            starting_step=starting_step,
            max_steps=max_steps,
            new_random_task_on_reset=self._new_random_task_on_reset,
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
        # We add a restriction to prevent users from getting data from
        # previous or future tasks.
        # TODO: Should the validation environment only be for the current task?
        starting_step = self.current_task_id * self.steps_per_task
        max_steps = starting_step + self.steps_per_task - 1
        return self._make_wrappers(
            base_env=self.val_dataset,
            task_schedule=self.valid_task_schedule,
            # sharp_task_boundaries=self.known_task_boundaries_at_train_time,
            task_labels_available=self.task_labels_at_train_time,
            transforms=self.val_transforms,
            starting_step=starting_step,
            max_steps=max_steps,
            new_random_task_on_reset=self._new_random_task_on_reset,
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
            transforms=self.test_transforms,
            starting_step=0,
            max_steps=self.max_steps,
            new_random_task_on_reset=self._new_random_task_on_reset,
        )

    def load_task_schedule(self, file_path: Path) -> Dict[int, Dict]:
        """ Load a task schedule from the given path. """
        with open(file_path) as f:
            task_schedule = json.load(f)
            return {int(k): task_schedule[k] for k in sorted(task_schedule.keys())}

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

        if is_classic_control_env(base_env):
            # If we are in a classic control env, and we dont want the state to
            # be fully-observable (i.e. we want pixel observations rather than
            # getting the pole angle, velocity, etc.), then add the
            # PixelObservation wrapper to the list of wrappers.
            if self.force_pixel_observations:
                wrappers.append(PixelObservationWrapper)

        elif is_atari_env(base_env):
            if isinstance(base_env, str) and "ram" in base_env:
                if self.force_pixel_observations:
                    raise NotImplementedError(
                        f"Can't force pixel observations when using the ram-version of "
                        f"an atari env!"
                    )
            else:
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

    def _temp_wrappers(self) -> List[Callable[[gym.Env], gym.Env]]:
        """ Gets the minimal wrappers needed to figure out the Spaces of the
        train/valid/test environments.

        This is called in the 'constructor' (__post_init__) to set the Setting's
        observation/action/reward spaces, so this should depend on as little
        state from `self` as possible, since not all attributes have been
        defined at the time when this is called.
        """
        return self._make_wrappers(
            base_env=self.dataset,
            task_schedule=self.train_task_schedule,
            # sharp_task_boundaries=self.known_task_boundaries_at_train_time,
            task_labels_available=self.task_labels_at_train_time,
            transforms=self.train_transforms,
            # These two shouldn't matter really:
            starting_step=0,
            max_steps=self.max_steps,
            new_random_task_on_reset=self._new_random_task_on_reset,
        )

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


class ContinualRLTestEnvironment(TestEnvironment, IterableWrapper):
    def __init__(self, *args, task_schedule: Dict, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_schedule = task_schedule
        self.boundary_steps = [
            step // (self.batch_size or 1) for step in self.task_schedule.keys()
        ]

    def __len__(self):
        return math.ceil(self.step_limit / (getattr(self.env, "batch_size", 1) or 1))

    def get_results(self) -> TaskSequenceResults[EpisodeMetrics]:
        # TODO: Place the metrics in the right 'bin' at the end of each episode during
        # testing depending on the task at that time, rather than what's happening here,
        # where we're getting all the rewards and episode lengths at the end and then
        # sort it out into the bins based on the task schedule. ALSO: this would make it
        # easier to support monitoring batched RL environments, since these `Monitor`
        # methods (get_episode_rewards, get_episode_lengths, etc) assume the environment
        # isn't batched.
        rewards = self.get_episode_rewards()
        lengths = self.get_episode_lengths()

        task_schedule: Dict[int, Dict] = self.task_schedule
        task_steps = sorted(task_schedule.keys())
        assert 0 in task_steps
        import bisect

        nb_tasks = len(task_steps)
        assert nb_tasks >= 1

        test_results = TaskSequenceResults(TaskResults() for _ in range(nb_tasks))

        # TODO: Fix this, since the task id might not be related to the steps!
        for step, episode_reward, episode_length in zip(
            itertools.accumulate(lengths), rewards, lengths
        ):
            # Given the step, find the task id.
            task_id = bisect.bisect_right(task_steps, step) - 1
            episode_metric = EpisodeMetrics(
                n_samples=1,
                mean_episode_reward=episode_reward,
                mean_episode_length=episode_length,
            )
            test_results[task_id].append(episode_metric)

        return test_results

    def render(self, mode="human", **kwargs):
        # TODO: This might not be setup right. Need to check.
        image_batch = super().render(mode=mode, **kwargs)
        if mode == "rgb_array" and self.batch_size:
            return tile_images(image_batch)
        return image_batch

    def _after_reset(self, observation):
        # Is this going to work fine when the observations are batched though?
        return super()._after_reset(observation)


if __name__ == "__main__":
    ContinualRLSetting.main()
