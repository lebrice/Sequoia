import itertools
import json
import math
import warnings
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, List, Optional, Sequence, Type, Union
import wandb
import gym
from gym import spaces
from gym.envs.registration import registry, load, spec, EnvRegistry, EnvSpec
from gym.utils import colorize
from gym.wrappers import TimeLimit
from sequoia.common import Config
from sequoia.common.gym_wrappers import (
    AddDoneToObservation,
    MultiTaskEnvironment,
    SmoothTransitions,
    TransformObservation,
    RenderEnvWrapper,
)
from sequoia.utils.utils import flag
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
from sequoia.settings.assumptions.continual import ContinualAssumption
from sequoia.settings.assumptions.continual import ContinualResults, TestEnvironment
from sequoia.settings.base import Method
from sequoia.utils import get_logger
from simple_parsing import choice, field, list_field
from simple_parsing.helpers import dict_field
from stable_baselines3.common.atari_wrappers import AtariWrapper

from sequoia.settings.rl import ActiveEnvironment
from .environment import GymDataLoader
from .make_env import make_batched_env
from .results import RLResults
from sequoia.settings.rl.wrappers import HideTaskLabelsWrapper, TypedObjectsWrapper
from sequoia.settings.rl.wrappers import MeasureRLPerformanceWrapper

from .objects import (
    Observations,
    ObservationType,
    Actions,
    ActionType,
    Rewards,
    RewardType,
)

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

# TODO: Create a fancier class for the TaskSchedule, as described in the test file.
TaskSchedule = Dict[int, Union[Dict, Callable[[gym.Env], None], Any]]


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
    Results: ClassVar[Type[RLResults]] = RLResults

    # Class variable that holds the dict of available environments.
    # TODO: There is a bit of duplicated code between this here and the callables in
    # `tasks.py`.. Is there perhaps a way to automatically create this dict from there?
    
    # TODO: Rework this, because for instance if we're given 'CartPole-v1', but
    # self.available_datasets has {"cartpole": "CartPole-v0"}, then we could just use
    # our multi-task wrapper on top of CartPole-v1, since it's going to work the same
    # way as if it were applied to CartPole-v0.
    available_datasets: ClassVar[Dict[str, str]] = {
        "cartpole": "CartPole-v0",
        "pendulum": "Pendulum-v0",
        "mountaincar": "MountainCar-v0",
        "acrobot": "Acrobot-v1",
        # "breakout": "Breakout-v0",
        # "duckietown": "Duckietown-straight_road-v0"
        "half_cheetah": "ContinualHalfCheetah-v2",
        "hopper": "ContinualHopper-v2",
        "walker2d": "ContinualWalker2d-v2",
        # "HalfCheetah-v2": "ContinualHalfCheetah-v2",
        # "Hopper-v2": "ContinualHopper-v2",
        # "Walker2d-v2": "ContinualWalker2d-v2",
    }
    # TODO: Add breakout to 'available_datasets' only when atari_py is installed.

    # Which environment (a.k.a. "dataset") to learn on.
    # The dataset could be either a string (env id or a key from the
    # available_datasets dict), a gym.Env, or a callable that returns a
    # single environment.
    dataset: str = choice(available_datasets, default="cartpole")
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

    # Deprecated: use `train_max_steps` instead.
    max_steps: Optional[int] = field(default=None, cmd=False)
    # Deprecated: use `train_max_steps` instead.
    test_steps: Optional[int] = field(default=None, cmd=False)

    def __post_init__(self):
        super().__post_init__()
        
        self.dataset = self._get_simple_name(self.dataset)
        if self.dataset not in self.available_datasets:
            raise RuntimeError(
                f"The chosen dataset/environment ({self.dataset}) isn't in the dict of "
                f"available datasets/environments, and a task schedule was not passed, "
                f"so this Setting {type(self).__name__} doesn't know how to create "
                f"tasks for that env!\n"
                f"Supported envs:\n"
                + ("\n".join(f"- {k}: {v}" for k, v in self.available_datasets.items()))
            )

        # The 'simple' names of the train/valid/test environments.
        self.train_dataset: str = self.train_dataset or self.dataset
        self.val_dataset: str = self.val_dataset or self.dataset
        self.test_dataset: str = self.test_dataset or self.dataset
                
        # The environment 'ID' associated with each 'simple name'.
        self.train_dataset_id: str = self._get_dataset_id(self.train_dataset)
        self.val_dataset_id: str = self._get_dataset_id(self.val_dataset)
        self.train_dataset_id: str = self._get_dataset_id(self.train_dataset)

        # Set the number of tasks depending on the increment, and vice-versa.
        # (as only one of the two should be used).
        assert self.train_max_steps, "assuming this should always be set, for now."

        # Load the task schedules from the corresponding files, if present.
        if self.train_task_schedule_path:
            self.train_task_schedule = _load_task_schedule(
                self.train_task_schedule_path
            )
        if self.val_task_schedule_path:
            self.val_task_schedule = _load_task_schedule(self.val_task_schedule_path)
        if self.test_task_schedule_path:
            self.test_task_schedule = _load_task_schedule(self.test_task_schedule_path)

        self.train_env: gym.Env
        self.valid_env: gym.Env
        self.test_env: gym.Env

        # Temporary environments which are created and used only for creating the task
        # schedules and closed right after.
        self._temp_train_env: Optional[gym.Env] = self._make_env(self.train_dataset_id)
        self._temp_val_env: Optional[gym.Env] = None
        self._temp_test_env: Optional[gym.Env] = None
        # Create the task schedules, using the 'task sampling' function from `tasks.py`.
        if not self.train_task_schedule:
            self.train_task_schedule = self.create_train_task_schedule()
        if not self.val_task_schedule:
            # Avoid creating an additional env, just reuse the train_temp_env.
            self._temp_val_env = (
                self._temp_train_env
                if self.val_dataset == self.train_dataset
                else self._make_env(self.val_dataset_id)
            )
            self.val_task_schedule = self.create_val_task_schedule()
        if not self.test_task_schedule:
            self._temp_test_env = (
                self._temp_train_env
                if self.test_dataset == self.train_dataset
                else self._make_env(self.val_dataset_id)
            )
            self.test_task_schedule = self.create_test_task_schedule()

        self._observation_space = self._temp_train_env.observation_space
        self._action_space = self._temp_train_env.observation_space
        self._reward_space = self._temp_train_env.reward_space  # TODO: Fix this.
        
        if self._temp_train_env:
            self._temp_train_env.close()
        if self._temp_val_env and self._temp_val_env is not self._temp_train_env:
            self._temp_val_env.close()
        if self._temp_test_env and self._temp_test_env is not self._temp_train_env:
            self._temp_test_env.close()

    # def get_env_class(self, env: str) -> Type[gym.Env]:
    #     """ Returns the class associated with the given environment name or env id. """
    #     env_id: str = ""
    #     if env in self.available_datasets:
    #         env_id = self.available_datasets[env]
    #     elif isinstance(env, str):
    #         env_id = env
    #     from gym.envs.registration import registry, load, spec, EnvRegistry, EnvSpec
    #     env_spec: EnvSpec = spec(env_id)
    #     entry_point = env_spec.entry_point
    #     return load(entry_point)

    def create_train_task_schedule(self) -> TaskSchedule:
        temp_env: gym.Env
        return {}

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
        if make_task_for_env.dispatch(
            type(env.unwrapped)
        ) is make_task_for_env.dispatch(object):
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

        # Run the Training loop (which is defined in IncrementalAssumption).
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
        if wandb.run:
            test_dir = wandb.run.dir
        else:
            test_dir = "results"
        # TODO: Debug wandb Monitor integration.
        self.test_env = ContinualRLTestEnvironment(
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
        return self._make_wrappers(
            base_env=self.dataset,
            task_schedule=self.train_task_schedule,
            # TODO: Removing this, but we have to check that it doesn't change when/how
            # the task boundaries are given to the Method.
            # sharp_task_boundaries=self.known_task_boundaries_at_train_time,
            task_labels_available=self.task_labels_at_train_time,
            transforms=self.train_transforms,
            starting_step=0,
            max_steps=self.max_steps,
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
            task_schedule=self.valid_task_schedule,
            # sharp_task_boundaries=self.known_task_boundaries_at_train_time,
            task_labels_available=self.task_labels_at_train_time,
            transforms=self.val_transforms,
            starting_step=0,
            max_steps=self.max_steps,
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
            transforms=self.test_transforms,
            starting_step=0,
            max_steps=self.test_steps,
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
            new_random_task_on_reset=self.stationary_context,
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

    def _get_simple_name(self, env_name_or_id: str) -> str:
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
        
        # if isinstance(env_name_or_id, str) and "-v" in env_name_or_id:
        #     env_name, _, env_version = env_name_or_id.partition("-v")
        #     # If there is a matching environment in the available datasets.
        #     warnings.warn(UserWarning(
        #         f"The given environment ID {env_name_or_id}"
        #     ))

        # return None
        # # if not isinstance(env_name_or_id, str):
        # raise NotImplementedError(
        #     f"Don't know how to get the 'simple name' of the given env name or id "
        #     f"{env_name_or_id}."
        # )
        # raise NotImplementedError()
    
    def _get_environment_id(
        self, env_name: Union[str, gym.Env, Callable[[], gym.Env]],
    ) -> str:
        """ Returns the environment 'id' to use for the given 'simple' name. """
        if env_name in self.available_datasets.keys():
            return self.available_datasets[env_name]

        if env_name in self.available_datasets.values():
            simple_name: str = [
                k for k, v in self.available_datasets.items() if v == env_name
            ][0]
            warnings.warn(
                DeprecationWarning(
                    f"`dataset` should be the simple name, rather than the environment ID! "
                    f"(Received {env_name}, instead of {simple_name})."
                )
            )
            return env_name
        if not isinstance(env_name, str):
            raise RuntimeError(
                f"WIP: Removing support for passing a custom callable or gym env!"
            )
        raise RuntimeError(
            f"The chosen dataset/environment ({env_name}) isn't in the dict of "
            f"available datasets/environments so this Setting ({type(self).__name__}) "
            f"doesn't know how to create tasks for that env!\n"
            f"Supported envs: \n"
            + ("\n".join(f"- {k}: {v}" for k, v in self.available_datasets.items()))
        )

        # raise NotImplementedError(
        #     "WIP: Removing support for passing a custom callable or gym env."
        # )
        # if isinstance(dataset, gym.Env) and batch_size:
        #     raise RuntimeError(
        #         "Batch size should be None when a gym.Env "
        #         "object is passed as `dataset`."
        #     )
        # if not isinstance(dataset, (str, gym.Env)) and not callable(dataset):
        #     raise RuntimeError(
        #         f"`dataset` must be either a string, a gym.Env, or a callable. "
        #         f"(got {dataset})"
        #     )


def _load_task_schedule(file_path: Path) -> Dict[int, Dict]:
    """ Load a task schedule from the given path. """
    with open(file_path) as f:
        task_schedule = json.load(f)
        return {int(k): task_schedule[k] for k in sorted(task_schedule.keys())}


class ContinualRLTestEnvironment(TestEnvironment, IterableWrapper):
    def __init__(self, *args, task_schedule: Dict, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_schedule = task_schedule
        self.boundary_steps = [
            step // (self.batch_size or 1) for step in self.task_schedule.keys()
        ]

    def __len__(self):
        return math.ceil(self.step_limit / (getattr(self.env, "batch_size", 1) or 1))

    def get_results(self) -> ContinualResults[EpisodeMetrics]:
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

        test_results = ContinualResults()
        # TODO: Fix this, since the task id might not be related to the steps!
        for step, episode_reward, episode_length in zip(
            itertools.accumulate(lengths), rewards, lengths
        ):
            # Given the step, find the task id.
            episode_metric = EpisodeMetrics(
                n_samples=1,
                mean_episode_reward=episode_reward,
                mean_episode_length=episode_length,
            )
            test_results.append(episode_metric)
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
