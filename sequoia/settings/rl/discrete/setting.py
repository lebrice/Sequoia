import itertools
import math
import warnings
from dataclasses import InitVar, dataclass, fields
from typing import Any, Callable, ClassVar, Dict, List, Optional, Union, Type

import gym
import numpy as np
from sequoia.common.gym_wrappers import IterableWrapper
from sequoia.common.gym_wrappers.batch_env.tile_images import tile_images
from sequoia.common.gym_wrappers.utils import is_monsterkong_env
from sequoia.common.metrics.rl_metrics import EpisodeMetrics
from sequoia.settings.assumptions.context_discreteness import DiscreteContextAssumption
from sequoia.settings.assumptions.incremental import TaskResults, TaskSequenceResults
from sequoia.settings.rl.envs import MUJOCO_INSTALLED
from sequoia.utils.logging_utils import get_logger
from sequoia.utils.utils import dict_union, pairwise
from simple_parsing import field
from simple_parsing.helpers import choice

from ..continual.setting import (
    ContinualRLSetting,
    ContinualRLTestEnvironment,
    supported_envs as _parent_supported_envs,
)
from .tasks import DiscreteTask, TaskSchedule, is_supported, make_discrete_task
from .tasks import registry, EnvSpec
from .test_environment import DiscreteTaskAgnosticRLTestEnvironment, TestEnvironment

from sequoia.settings.rl.envs import MONSTERKONG_INSTALLED
logger = get_logger(__file__)

supported_envs: Dict[str, EnvSpec] = dict_union(
    _parent_supported_envs,
    {
        spec.id: spec
        for env_id, spec in registry.env_specs.items()
        if spec.id not in _parent_supported_envs and is_supported(env_id)
    },
)
available_datasets: Dict[str, str] = {env_id: env_id for env_id in supported_envs}

from .results import DiscreteTaskAgnosticRLResults
from sequoia.settings.base import Results



@dataclass
class DiscreteTaskAgnosticRLSetting(DiscreteContextAssumption, ContinualRLSetting):
    """ Continual Reinforcement Learning Setting where there are clear task boundaries,
    but where the task information isn't available.
    """
    # TODO: Update the type or results that we get for this Setting.
    Results: ClassVar[Type[Results]] = DiscreteTaskAgnosticRLResults
    
    
    
    # The type wrapper used to wrap the test environment, and which produces the
    # results.
    TestEnvironment: ClassVar[
        Type[TestEnvironment]
    ] = DiscreteTaskAgnosticRLTestEnvironment

    # The function used to create the tasks for the chosen env.
    _task_sampling_function: ClassVar[Callable[..., DiscreteTask]] = make_discrete_task

    # Class variable that holds the dict of available environments.
    available_datasets: ClassVar[Dict[str, Union[str, Any]]] = available_datasets

    # Which environment (a.k.a. "dataset") to learn on.
    # The dataset could be either a string (env id or a key from the
    # available_datasets dict), a gym.Env, or a callable that returns a
    # single environment.
    dataset: str = choice(available_datasets, default="CartPole-v0")

    # The number of "tasks" that will be created for the training, valid and test
    # environments. When left unset, will use a default value that makes sense
    # (something like 5).
    nb_tasks: int = field(5, alias=["n_tasks", "num_tasks"])

    # Maximum number of training steps per task.
    train_steps_per_task: Optional[int] = None
    # Number of test steps per task.
    test_steps_per_task: Optional[int] = None

    # # Maximum number of episodes in total.
    # train_max_episodes: Optional[int] = None
    # # TODO: Add tests for this 'max episodes' and 'episodes_per_task'.
    # train_max_episodes_per_task: Optional[int] = None
    # # Total number of steps in the test loop. (Also acts as the "length" of the testing
    # # environment.)
    # test_max_steps_per_task: int = 10_000
    # test_max_episodes_per_task: Optional[int] = None

    # # Max number of steps per training task. When left unset and when `train_max_steps`
    # # is set, takes the value of `train_max_steps` divided by `nb_tasks`.
    # train_max_steps_per_task: Optional[int] = None
    # # (WIP): Maximum number of episodes per training task. When left unset and when
    # # `train_max_episodes` is set, takes the value of `train_max_episodes` divided by
    # # `nb_tasks`.
    # train_max_episodes_per_task: Optional[int] = None
    # # Maximum number of steps per task in the test loop. When left unset and when
    # # `test_max_steps` is set, takes the value of `test_max_steps` divided by `nb_tasks`.
    # test_max_steps_per_task: Optional[int] = None
    # # (WIP): Maximum number of episodes per test task. When left unset and when
    # # `test_max_episodes` is set, takes the value of `test_max_episodes` divided by
    # # `nb_tasks`.
    # test_max_episodes_per_task: Optional[int] = None

    # def warn(self, warning: Warning):
    #     logger.warning(warning)
    #     warnings.warn(warning)

    def __post_init__(self):
        # TODO: Rework all the messy fields from before by just considering these as eg.
        # the maximum number of steps per task, rather than the fixed number of steps
        # per task.
        assert not self.smooth_task_boundaries

        super().__post_init__()

        if self.max_episode_steps is None:
            if is_monsterkong_env(self.dataset):
                self.max_episode_steps = 500

    def create_train_task_schedule(self) -> TaskSchedule[DiscreteTask]:
        # IDEA: Could convert max_episodes into max_steps if max_steps_per_episode is
        # set.
        return super().create_train_task_schedule()

    def create_val_task_schedule(self) -> TaskSchedule[DiscreteTask]:
        # Always the same as train task schedule for now.
        return super().create_val_task_schedule()

    def create_test_task_schedule(self) -> TaskSchedule[DiscreteTask]:
        return super().create_test_task_schedule()
