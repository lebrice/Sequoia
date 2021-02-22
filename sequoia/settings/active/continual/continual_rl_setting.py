import itertools
import json
import operator
import warnings
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, ClassVar, Dict, List, Optional, Sequence, Type, Union

import gym
import numpy as np
from gym import spaces
from gym.envs.classic_control import CartPoleEnv
from gym.utils import colorize
from gym.wrappers import AtariPreprocessing, TimeLimit
from simple_parsing import choice, field, list_field
from simple_parsing.helpers import dict_field
from stable_baselines3.common.atari_wrappers import AtariWrapper
from torch import Tensor

from sequoia.common import Batch, Config, Metrics
from sequoia.common.gym_wrappers import (
    AddDoneToObservation,
    AddInfoToObservation,
    MultiTaskEnvironment,
    SmoothTransitions,
    TransformAction,
    TransformObservation,
    TransformReward,
)
from sequoia.common.gym_wrappers.batch_env import (
    BatchedVectorEnv,
    SyncVectorEnv,
    VectorEnv,
)
from sequoia.common.gym_wrappers.batch_env.tile_images import tile_images
from sequoia.common.gym_wrappers.env_dataset import EnvDataset
from sequoia.common.gym_wrappers.pixel_observation import (
    ImageObservations,
    PixelObservationWrapper,
)
from sequoia.common.gym_wrappers.step_callback_wrapper import StepCallbackWrapper
from sequoia.common.gym_wrappers.utils import (
    IterableWrapper,
    classic_control_env_prefixes,
    classic_control_envs,
    has_wrapper,
    is_atari_env,
    is_classic_control_env,
)
from sequoia.common.metrics import RegressionMetrics
from sequoia.common.spaces import Image, Sparse
from sequoia.common.spaces.named_tuple import NamedTuple, NamedTupleSpace
from sequoia.common.transforms import Transforms
from sequoia.settings.active import ActiveSetting
from sequoia.settings.assumptions.incremental import IncrementalSetting, TestEnvironment
from sequoia.settings.base import Method
from sequoia.settings.base.results import Results

from sequoia.utils import dict_union, get_logger
from .. import ActiveEnvironment
from .gym_dataloader import GymDataLoader
from .make_env import make_batched_env
from .rl_results import RLResults
from .wrappers import (
    HideTaskLabelsWrapper,
    NoTypedObjectsWrapper,
    RemoveTaskLabelsWrapper,
    TypedObjectsWrapper,
)

from sequoia.settings.assumptions.incremental import TaskSequenceResults, TaskResults
from sequoia.common.metrics.rl_metrics import EpisodeMetrics

logger = get_logger(__file__)

# TODO: Implement a get_metrics (ish) in the Environment, not on the Setting!
# TODO: The validation environment will also call the on_task_switch when it
# reaches a task boundary, and there isn't currently a way to distinguish if
# that method is being called because of the training or because of the
# validation environment.


task_params: Dict[Union[Type[gym.Env], str], List[str]] = {
    "CartPole-v0": [
        "gravity",  #: 9.8,
        "masscart",  #: 1.0,
        "masspole",  #: 0.1,
        "length",  #: 0.5,
        "force_mag",  #: 10.0,
        "tau",  #: 0.02,
    ],
    # TODO: Add more of the classic control envs here.
    # TODO: Need to get the attributes to modify in each environment type and
    # add them here.
    # AtariEnv: [
    #     # TODO: Maybe have something like the difficulty as the CL 'task' ?
    #     # difficulties = temp_env.ale.getAvailableDifficulties()
    #     # "game_difficulty",
    # ],
}

# Type alias for the Environment returned by `train/val/test_dataloader`.
Environment = ActiveEnvironment[
    "ContinualRLSetting.Observations",
    "ContinualRLSetting.Observations",
    "ContinualRLSetting.Rewards",
]
from sequoia.settings.assumptions.incremental import (
    TaskResults,
    TaskSequenceResults,
    IncrementalResults,
)
from sequoia.common.metrics.rl_metrics import EpisodeMetrics


@dataclass
class ContinualRLSetting(ActiveSetting, IncrementalSetting):
    """ Reinforcement Learning Setting where the environment changes over time.

    This is an Active setting which uses gym environments as sources of data.
    These environments' attributes could change over time following a task
    schedule. An example of this could be that the gravity increases over time
    in cartpole, making the task progressively harder as the agent interacts with
    the environment.
    """

    # The type of results returned by an RL experiment.
    Results: ClassVar[Type[Results]] = RLResults

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
    }
    # TODO: Add breakout to 'available_datasets' only when atari_py is installed.

    # Which environment (a.k.a. "dataset") to learn on.
    # The dataset could be either a string (env id or a key from the
    # available_datasets dict), a gym.Env, or a callable that returns a single environment.
    # If self.dataset isn't one of those, an error will be raised.
    dataset: str = choice(available_datasets, default="cartpole")

    # The number of tasks. By default 1 for this setting.
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

    # Number of steps per task in the test loop.
    test_steps_per_task: int = 10_000
    # Total number of steps in the test loop. By default, takes the value of
    # `test_steps_per_task * nb_tasks`.
    test_steps: Optional[int] = None

    # Standard deviation of the multiplicative Gaussian noise that is used to
    # create the values of the env attributes for each task.
    task_noise_std: float = 0.2

    # Wether the task boundaries are smooth or sudden.
    smooth_task_boundaries: bool = True

    # Wether to observe the state directly, rather than pixels. This can be
    # useful to debug environments like CartPole, for instance.
    observe_state_directly: bool = False

    # Path to a json file from which to read the train task schedule.
    train_task_schedule_path: Optional[Path] = None
    # Path to a json file from which to read the validation task schedule.
    valid_task_schedule_path: Optional[Path] = None
    # Path to a json file from which to read the test task schedule.
    test_task_schedule_path: Optional[Path] = None

    # NOTE: Added this `cmd=False` option to mark that we don't want to generate
    # any command-line arguments for these fields.
    train_task_schedule: Dict[int, Dict[str, float]] = dict_field(cmd=False)
    valid_task_schedule: Dict[int, Dict[str, float]] = dict_field(cmd=False)
    test_task_schedule: Dict[int, Dict[str, float]] = dict_field(cmd=False)

    train_wrappers: List[Callable[[gym.Env], gym.Env]] = list_field(cmd=False)
    valid_wrappers: List[Callable[[gym.Env], gym.Env]] = list_field(cmd=False)
    test_wrappers: List[Callable[[gym.Env], gym.Env]] = list_field(cmd=False)

    # Wether observations from the environments whould include
    # the end-of-episode signal. Only really useful if your method will iterate
    # over the environments in the dataloader style
    # (as does the baseline method).
    add_done_to_observations: bool = False

    batch_size: Optional[int] = field(default=None, cmd=False)
    num_workers: Optional[int] = field(default=None, cmd=False)

    # The maximum number of steps per episode. When None, there is no limit.
    max_episode_steps: Optional[int] = None

    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)
        self._new_random_task_on_reset: bool = False

        # Post processing of the 'dataset' field:
        if self.dataset in self.available_datasets.keys():
            # the environment name was passed, rather than an id
            # (e.g. 'cartpole' -> 'CartPole-v0").
            self.dataset = self.available_datasets[self.dataset]

        elif self.dataset not in self.available_datasets.values():
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
                f"Batch size should be None when a gym.Env "
                f"object is passed as `dataset`."
            )
        if not isinstance(self.dataset, (str, gym.Env)) and not callable(self.dataset):
            raise RuntimeError(
                f"`dataset` must be either a string, a gym.Env, or a callable. "
                f"(got {self.dataset})"
            )

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
            # A task schedule was passed: infer the number of tasks from it.
            change_steps = sorted(self.train_task_schedule.keys())
            assert 0 in change_steps, "Schedule needs a task at step 0."

            # NOTE: When in a ContinualRLSetting with smooth task boundaries,
            # the last entry in the schedule represents the state of the env at
            # the end of the "task". When there are clear task boundaries (i.e.
            # when in 'Class'/Task-Incremental RL), the last entry is the start
            # of the last task.
            if not self.smooth_task_boundaries:
                self.nb_tasks = len(change_steps)

            # TODO: @lebrice: I guess we have to assume that the interval
            # between steps is constant for now? Do we actually depend on this
            # being the case? I think steps_per_task is only really ever used
            # for creating the task schedule, which we already have in this
            # case.
            assert (
                len(change_steps) >= 2
            ), "WIP: need a minimum of two tasks in the task schedule for now."
            self.steps_per_task = change_steps[1] - change_steps[0]

            for i in range(len(change_steps) - 1):
                if change_steps[i + 1] - change_steps[i] != self.steps_per_task:
                    raise NotImplementedError(
                        f"WIP: This might not work yet if the tasks aren't "
                        f"equally spaced out at a fixed interval."
                    )
            self.max_steps = max(change_steps)
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

        if not all([self.nb_tasks, self.max_steps, self.steps_per_task]):
            raise RuntimeError(
                f"You need to provide at least two of 'max_steps', "
                f"'nb_tasks', or 'steps_per_task'."
            )

        if not self.smooth_task_boundaries:
            assert self.nb_tasks == self.max_steps // self.steps_per_task

        if self.test_task_schedule:
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
                        f"WIP: This might not work yet if the test tasks aren't "
                        f"equally spaced out at a fixed interval."
                    )
            self.test_steps = max(change_steps)
            if not self.smooth_task_boundaries:
                # See above note about the last entry.
                self.test_steps += self.test_steps_per_task
        elif self.test_steps is None:
            assert (
                self.test_steps_per_task
            ), "need to set one of test_steps or test_steps_per_task"
            self.test_steps = self.test_steps_per_task * self.nb_tasks
        else:
            assert (
                self.test_steps
            ), "need to set one of test_steps or test_steps_per_task"
            self.test_steps_per_task = self.test_steps // self.nb_tasks

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
            self.nb_tasks = 1
            self.steps_per_task = self.max_steps

        # Task schedules for training / validation and testing.

        # Create a temporary environment so we can extract the spaces and create
        # the task schedules.
        with self._make_env(
            self.dataset, self._temp_wrappers(), self.observe_state_directly
        ) as temp_env:
            # FIXME: Replacing the observation space dtypes from their original
            # 'generated' NamedTuples to self.Observations. The alternative
            # would be to add another argument to the MultiTaskEnv wrapper, to
            # pass down a dtype to be set on its observation_space's `dtype`
            # attribute, which would be ugly.
            assert isinstance(temp_env.observation_space, NamedTupleSpace)
            temp_env.observation_space.dtype = self.Observations
            # Populate the task schedules created above.
            if not self.train_task_schedule:
                train_change_steps = list(range(0, self.max_steps, self.steps_per_task))
                if self.smooth_task_boundaries:
                    # Add a last 'task' at the end of the 'epoch', so that the
                    # env changes smoothly right until the end.
                    train_change_steps.append(self.max_steps)
                self.train_task_schedule = self.create_task_schedule(
                    temp_env, train_change_steps,
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

            # Set the spaces using the temp env.
            self.observation_space = temp_env.observation_space
            self.action_space = temp_env.action_space
            self.reward_range = temp_env.reward_range
            self.reward_space = getattr(
                temp_env,
                "reward_space",
                spaces.Box(
                    low=self.reward_range[0], high=self.reward_range[1], shape=()
                ),
            )
            
        del temp_env

        self.train_env: gym.Env
        self.valid_env: gym.Env
        self.test_env: gym.Env

    def create_task_schedule(
        self, temp_env: MultiTaskEnvironment, change_steps: List[int]
    ) -> Dict[int, Dict]:
        """ Create the task schedule, which maps from a step to the changes that
        will occur in the environment when that step is reached.
        
        Uses the provided `temp_env` to generate the random tasks at the steps
        given in `change_steps` (a list of integers).

        Returns a dictionary mapping from integers (the steps) to the changes
        that will occur in the env at that step.

        TODO: IDEA: Instead of just setting env attributes, use the
        `methodcaller` or `attrsetter` from the `operator` built-in package,
        that way later when we want to add support for Meta-World, we can just
        use `partial(methodcaller("set_task"), task="new_task")(env)` or
        something like that (i.e. generalize from changing an attribute to
        applying a function on the env, which would allow calling methods in
        addition to setting attributes.)
        """
        task_schedule: Dict[int, Dict] = {}
        # Start with the default task (step 0) and then add a new task at
        # intervals of `self.steps_per_task`
        for task_step in change_steps:
            if task_step == 0:
                # Start with the default task, so that we can recover the 'iid'
                # case with standard env dynamics when there is only one task
                # and no non-stationarity.
                task_schedule[task_step] = temp_env.default_task
            else:
                task_schedule[task_step] = temp_env.random_task()

        return task_schedule

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
            logger.debug(f"Parsing the Config from the command-line.")
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
                f"Train tasks: "
                + json.dumps(list(self.train_task_schedule.values()), indent="\t")
            )
        else:
            logger.info(
                f"Train task schedule:"
                + json.dumps(self.train_task_schedule, indent="\t")
            )
        if self.config.debug:
            logger.debug(
                f"Test task schedule:"
                + json.dumps(self.test_task_schedule, indent="\t")
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
            observe_state_directly=self.observe_state_directly,
        )
        env_dataloader = self._make_env_dataloader(
            env_factory,
            batch_size=batch_size,
            num_workers=num_workers,
            max_steps=self.steps_per_task,
            max_episodes=self.episodes_per_task,
        )
        self.train_env = env_dataloader
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
            base_env=self.dataset,
            wrappers=self.valid_wrappers,
            observe_state_directly=self.observe_state_directly,
        )
        env_dataloader = self._make_env_dataloader(
            env_factory,
            batch_size=batch_size or self.batch_size,
            num_workers=num_workers if num_workers is not None else self.num_workers,
            max_steps=self.steps_per_task,
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
            base_env=self.dataset,
            wrappers=self.test_wrappers,
            observe_state_directly=self.observe_state_directly,
        )
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

    @staticmethod
    def _make_env(
        base_env: Union[str, gym.Env, Callable[[], gym.Env]],
        wrappers: List[Callable[[gym.Env], gym.Env]] = None,
        observe_state_directly: bool = False,
    ) -> gym.Env:
        """ Helper function to create a single (non-vectorized) environment. """
        env: gym.Env
        if isinstance(base_env, str):
            if base_env.startswith("MetaMonsterKong") and observe_state_directly:
                env = gym.make(base_env, observe_state=True)
            else:
                env = gym.make(base_env)
        elif isinstance(base_env, gym.Env):
            env = base_env
        elif callable(base_env):
            env = base_env()
        else:
            raise RuntimeError(
                f"base_env should either be a string, a callable, or a gym "
                f"env. (got {base_env})."
            )
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

        ## Apply the "post-batch" wrappers:
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
            task_schedule=self.train_task_schedule,
            sharp_task_boundaries=self.known_task_boundaries_at_train_time,
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
            task_schedule=self.valid_task_schedule,
            sharp_task_boundaries=self.known_task_boundaries_at_train_time,
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
            task_schedule=self.test_task_schedule,
            sharp_task_boundaries=self.known_task_boundaries_at_test_time,
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
        task_schedule: Dict[int, Dict],
        sharp_task_boundaries: bool,
        task_labels_available: bool,
        transforms: List[Transforms],
        starting_step: int,
        max_steps: int,
        new_random_task_on_reset: bool,
    ) -> List[Callable[[gym.Env], gym.Env]]:
        """ helper function for creating the train/valid/test wrappers. 
        
        These wrappers get applied *before* the batching, if applicable.
        """
        wrappers: List[Callable[[gym.Env], gym.Env]] = []
        # NOTE: When transitions are smooth, there are no "task boundaries".
        assert sharp_task_boundaries == (not self.smooth_task_boundaries)

        # TODO: Add some kind of Wrapper around the dataset to make it
        # semi-supervised?

        if self.max_episode_steps:
            wrappers.append(
                partial(TimeLimit, max_episode_steps=self.max_episode_steps)
            )

        if is_classic_control_env(self.dataset) and not self.observe_state_directly:
            # If we are in a classic control env, and we dont want the state to
            # be fully-observable (i.e. we want pixel observations rather than
            # getting the pole angle, velocity, etc.), then add the
            # PixelObservation wrapper to the list of wrappers.
            wrappers.append(PixelObservationWrapper)
            wrappers.append(ImageObservations)

        if (
            isinstance(self.dataset, str)
            and self.dataset.lower().startswith("metamonsterkong")
            and not self.observe_state_directly
        ):
            # TODO: Do we need the AtariPreprocessing wrapper on MonsterKong?
            # wrappers.append(partial(AtariPreprocessing, frame_skip=1))
            pass
        elif is_atari_env(self.dataset):
            # TODO: Test & Debug this: Adding the Atari preprocessing wrapper.
            # TODO: Figure out the differences (if there are any) between the
            # AtariWrapper from SB3 and the AtariPreprocessing wrapper from gym.
            wrappers.append(AtariWrapper)
            # wrappers.append(AtariPreprocessing)
            wrappers.append(ImageObservations)

        # Apply image transforms if the env will have image-like obs space
        if not self.observe_state_directly:
            # wrappers.append(ImageObservations)
            # Wrapper to apply the image transforms to the env.
            wrappers.append(partial(TransformObservation, f=transforms))

        # Add a wrapper which will add non-stationarity to the environment.
        # The "task" transitions will either be sharp or smooth.
        # In either case, the task ids for each sample are added to the
        # observations, and the dicts containing the task information (i.e. the
        # current values of the env attributes from the task schedule) get added
        # to the 'info' dicts.
        if sharp_task_boundaries:
            assert self.nb_tasks >= 1
            # Add a wrapper that creates sharp tasks.
            cl_wrapper = MultiTaskEnvironment
        else:
            # Add a wrapper that creates smooth tasks.
            cl_wrapper = SmoothTransitions

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
            task_schedule=self.train_task_schedule,
            sharp_task_boundaries=self.known_task_boundaries_at_train_time,
            task_labels_available=self.task_labels_at_train_time,
            transforms=self.train_transforms,
            # These two shouldn't matter really:
            starting_step=0,
            max_steps=self.max_steps,
            new_random_task_on_reset=self._new_random_task_on_reset,
        )


class ContinualRLTestEnvironment(TestEnvironment, IterableWrapper):
    def __init__(self, *args, task_schedule: Dict, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_schedule = task_schedule

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
        total_steps = self.get_total_steps()

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
