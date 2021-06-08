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


@dataclass
class DiscreteTaskAgnosticRLSetting(DiscreteContextAssumption, ContinualRLSetting):
    """ Continual Reinforcement Learning Setting where there are clear task boundaries,
    but where the task information isn't available.
    """

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

    # train_max_steps_per_task: int = 20_000
    # # Maximum number of episodes in total.
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
        defaults = {f.name: f.default for f in fields(self)}
        nb_tasks_given = self.nb_tasks != defaults["nb_tasks"]
        train_max_steps_given = self.train_max_steps != defaults["train_max_steps"]
        if self.train_task_schedule:
            # TODO: Handle the edge-case where train_max_steps_given will be False
            # because the train_max_steps from the task schedule is the same as what
            # we'd expect to receive (e.g. {0: <task0>, 100_000: <task1>}, default
            # value of train_max_steps is 100_000).
            pass
            # if not nb_tasks_given and not train_max_steps_given:
            #     raise RuntimeError(
            #         "One of 'nb_tasks' or 'train_max_steps' must be set when passing a "
            #         "task schedule."
            #     )
        # elif not nb_tasks_given:
        #     raise RuntimeError(
        #         "`nb_tasks` must be set when a task schedule isn't passed."
        #     )

        assert not self.smooth_task_boundaries

        super().__post_init__()

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

        # super().__post_init__()
        if self.max_episode_steps is None:
            if is_monsterkong_env(self.dataset):
                self.max_episode_steps = 500

    def create_train_task_schedule(self) -> TaskSchedule[DiscreteTask]:
        change_steps = np.linspace(
            0, self.train_max_steps, self.nb_tasks + 1, endpoint=True, dtype=int
        ).tolist()
        # IDEA: Could convert max_episodes into max_steps if max_steps_per_episode is
        # set.
        return self.create_task_schedule(
            temp_env=self._temp_train_env,
            change_steps=change_steps
            # # TODO: Add properties for the train/valid/test seeds?
            # seed=self.train_seed,
        )

    def create_val_task_schedule(self) -> TaskSchedule:
        # Always the same as train task schedule for now.
        return self.train_task_schedule.copy()
        # return self.create_task_schedule(
        #     temp_env=self._temp_val_env, change_steps=change_steps
        # )

    def create_test_task_schedule(self) -> TaskSchedule:
        n_boundaries = len(self.train_task_schedule)
        # Re-scale the steps in the task schedule based on self.test_max_steps
        # NOTE: Using the same task schedule as in training and validation for now.
        test_boundaries = np.linspace(0, self.test_max_steps, n_boundaries)
        return {
            step: task
            for step, task in zip(test_boundaries, self.train_task_schedule.values())
        }
        # change_steps = [0, self.test_max_steps]
        # return self.create_task_schedule(
        #     temp_env=self._temp_test_env,
        #     change_steps=change_steps,
        #     # seed=self.train_seed,
        # )
