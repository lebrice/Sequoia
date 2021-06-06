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
from .test_environment import DiscreteTaskAgnosticRLTestEnvironment

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
    TestEnvironment: ClassVar[Type[TestEnvironment]] = DiscreteTaskAgnosticRLTestEnvironment
    
    # The function used to create the tasks for the chosen env.
    _task_sampling_function: ClassVar[Callable[..., DiscreteTask]] = make_discrete_task

    # Class variable that holds the dict of available environments.
    available_datasets: ClassVar[Dict[str, Union[str, Any]]] = dict_union(
        available_datasets, {"monsterkong": "MetaMonsterKong-v0"},
    )

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
                    f"`train_max_steps` ({self.train_max_steps}). "
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

        # # assert self.train_max_steps == train_max_steps
        # if self.nb_tasks == nb_train_tasks - 1 or self.train_max_steps == train_last_boundary:
        #     # If the task schedule is 1-longer than we expect, then drop that
        #     # last entry. (This happens when using the same 'benchmark' yaml
        #     # file for both the ContinualRLSetting and this Discrete setting,
        #     # since in ContinualRLSetting the last entry in the task schedule
        #     # shows the task state at the end of training.
        #     warning = UserWarning(
        #         "Dropping the last entry in the passed train task schedule"
        #     )
        #     warnings.warn(warning)
        #     logger.warning(warning)
        #     self.train_task_schedule.pop(max(self.train_task_schedule.keys()))
        #     nb_train_tasks -= 1
        #     train_last_boundary = max(self.train_task_schedule.keys())

        # if self.nb_tasks == nb_test_tasks - 1 or self.test_max_steps == test_last_boundary:
        #     warning = UserWarning(
        #         "Dropping the last entry in the passed test task schedule"
        #     )
        #     warnings.warn(warning)
        #     logger.warning(warning)
        #     self.test_task_schedule.pop(max(self.test_task_schedule.keys()))
        #     nb_test_tasks -= 1
        #     test_last_boundary = max(self.test_task_schedule.keys())

        # assert self.nb_tasks == len(self.train_task_schedule), (self.nb_tasks, len(self.train_task_schedule))
        # assert self.nb_tasks == len(self.test_task_schedule), (self.nb_tasks, len(self.test_task_schedule))
        # assert self.train_max_steps = max(self.train_task_schedule) + task_lengths[-1]

        # elif self.nb_tasks != defaults["nb_tasks"]:

        # if self.nb_tasks == nb_test_tasks - 1:
        #     # If the task schedule is 1-longer than we expect, then drop that
        #     # last entry. (This happens when using the same 'benchmark' yaml
        #     # file for both the ContinualRLSetting and this Discrete setting,
        #     # since in ContinualRLSetting the last entry in the task schedule
        #     # shows the task state at the end of training.
        #     self.warn(
        #         RuntimeWarning(
        #             "Dropping the last entry in the passed test task schedule"
        #         )
        #     )
        #     self.test_task_schedule.pop(max(self.test_task_schedule.keys()))
        # elif self.nb_tasks != nb_test_tasks:
        #     raise RuntimeError(
        #         f"The passed number of tasks ({self.nb_tasks}) is inconsistent "
        #         f"with the given test task schedule, which has {nb_test_tasks} "
        #         f"tasks in it."
        #     )
        # else:
        #     assert all(task_length > 1 for task_length in train_task_lengths)
        #     train_max_steps = train_last_task_step + train_task_lengths[-1]

        #     if (
        #         self.train_max_steps != defaults["train_max_steps"]
        #         and self.train_max_steps != train_max_steps
        #     ):
        #         if self.train_max_steps == last_boundary_step:
        #             # TODO: Do we drop the last entry in the task schedule? or do we
        #             # change the max number of steps?
        #             # Opting for dropping the last entry for now.
        #             self.train_task_schedule.pop(max(self.train_task_schedule))
        #             # self.warn(UserWarning(
        #             #     f"Setting the max number of steps to {train_max_steps}, "
        #             #     f"rather than {self.train_max_steps}."
        #             # ))

        #             self.train_max_steps = train_max_steps
        #         # TODO: Could there be an off-by-1 error here due to the rounding?
        #         elif abs(self.train_max_steps - train_max_steps) <= 2:
        #             self.train_max_steps = train_max_steps
        #         else:
        #             raise RuntimeError(
        #                 f"Value of train_max_steps ({self.train_max_steps}) is "
        #                 f"inconsistent with the given train task schedule, which has "
        #                 f"the last task boundary at step {last_boundary_step}, with "
        #                 f"task lengths of {task_lengths}, as it suggests the maximum "
        #                 f"total number of steps to be {last_boundary_step} + "
        #                 f"{task_lengths[-1]} => {train_max_steps}!"
        #             )

        # if self.nb_tasks and self.nb_tasks != nb_tasks:
        #     # If a custom number of tasks was passed, and it is different from that
        #     # of the task schedule:

        #     else:

        # self.nb_tasks = len(self.train_task_schedule)

        # defaults = {f.name: f.default for f in fields(self)}

        # if self.train_max_steps_per_task:
        #     self.train_task_schedule = type(self.train_task_schedule)({
        #         i * self.train_max_steps_per_task: self.train_task_schedule[step]
        #         for i, step in enumerate(sorted(self.train_task_schedule.keys()))
        #     })

        #     self.train_max_steps = train_max_steps
        # if self.train_task_schedule:
        #     self.nb_tasks = len(self.train_task_schedule)
        #     assert not self.train_max_steps == max(self.train_task_schedule)
        # if self.train_max_steps

        # NOTE: Calling super().__post_init__() will create the task schedules (using
        # `create_[train/val/test]_task_schedule()`) so we the fields used in those need
        # to be set here before calling `super().__post_init__()`.
        # self._make_consistent_fields()
        # if not all([self.nb_tasks, self.train_max_steps, self.train_steps_per_task]):
        #     raise RuntimeError(
        #         "You need to provide at least two of 'max_steps', "
        #         "'nb_tasks', or 'steps_per_task'."
        #     )

        # assert self.nb_tasks == self.train_max_steps // self.train_steps_per_task, (
        #     self.nb_tasks,
        #     self.train_max_steps,
        #     self.train_steps_per_task,
        # )

        # super().__post_init__()
        if self.max_episode_steps is None:
            if is_monsterkong_env(self.dataset):
                self.max_episode_steps = 500

    def create_train_task_schedule(self) -> TaskSchedule[DiscreteTask]:
        # change_steps: List[int]
        # if self.train_max_steps_per_task:
        #     assert self.train_max_steps == self.nb_tasks * self.train_max_steps_per_task
        #     change_steps = list(
        #         range(0, self.train_max_steps, self.train_max_steps_per_task)
        #     )
        # else:
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

        change_steps = [0, self.test_max_steps]
        return self.create_task_schedule(
            temp_env=self._temp_test_env,
            change_steps=change_steps,
            # seed=self.train_seed,
        )

    # def _make_fields_consistent(self):
    #     # TODO: Clean this up, its not clear enough which options take precedence on
    #     # other options.
    #     if self.train_task_schedule:
    #         if self.train_steps_per_task is not None:
    #             # If steps per task was passed, then we overwrite the keys of the tasks
    #             # schedule.
    #             self.train_task_schedule = {
    #                 i * self.train_steps_per_task: self.train_task_schedule[step]
    #                 for i, step in enumerate(sorted(self.train_task_schedule.keys()))
    #             }
    #         else:
    #             # A task schedule was passed: infer the number of tasks from it.
    #             change_steps = sorted(self.train_task_schedule.keys())
    #             assert 0 in change_steps, "Schedule needs a task at step 0."
    #             # TODO: @lebrice: I guess we have to assume that the interval
    #             # between steps is constant for now? Do we actually depend on this
    #             # being the case? I think steps_per_task is only really ever used
    #             # for creating the task schedule, which we already have in this
    #             # case.
    #             assert (
    #                 len(change_steps) >= 2
    #             ), "WIP: need a minimum of two tasks in the task schedule for now."
    #             self.steps_per_task = change_steps[1] - change_steps[0]
    #             # Double-check that this is the case.
    #             for i in range(len(change_steps) - 1):
    #                 if change_steps[i + 1] - change_steps[i] != self.steps_per_task:
    #                     raise NotImplementedError(
    #                         "WIP: This might not work yet if the tasks aren't "
    #                         "equally spaced out at a fixed interval."
    #                     )

    #         nb_tasks = len(self.train_task_schedule)
    #         default_number_of_tasks: int = [
    #             f.default for f in fields(self) if f.name == "nb_tasks"
    #         ][0]
    #         assert False, (nb_tasks, default_number_of_tasks, self.nb_tasks)
    #         if self.nb_tasks != default_number_of_tasks:
    #             if self.nb_tasks != nb_tasks:
    #                 raise RuntimeError(
    #                     f"Passed number of tasks {self.nb_tasks} doesn't match the "
    #                     f"number of tasks deduced from the task schedule ({nb_tasks})"
    #                 )
    #         self.nb_tasks = nb_tasks
    #         # TODO: Sort out this mess:
    #         if self.train_max_steps != self.nb_tasks * self.steps_per_task:
    #             self.train_max_steps = max(self.train_task_schedule.keys())

    #         # See above note about the last entry.
    #         self.train_max_steps += self.steps_per_task

    #     elif self.nb_tasks:
    #         if self.steps_per_task:
    #             self.train_max_steps = self.nb_tasks * self.steps_per_task
    #         elif self.train_max_steps:
    #             self.steps_per_task = self.train_max_steps // self.nb_tasks

    #     elif self.steps_per_task:
    #         if self.nb_tasks:
    #             self.train_max_steps = self.nb_tasks * self.steps_per_task
    #         elif self.train_max_steps:
    #             self.nb_tasks = self.train_max_steps // self.steps_per_task

    #     elif self.train_max_steps:
    #         if self.nb_tasks:
    #             self.steps_per_task = self.train_max_steps // self.nb_tasks
    #         elif self.steps_per_task:
    #             self.nb_tasks = self.train_max_steps // self.steps_per_task
    #         else:
    #             self.nb_tasks = 1
    #             self.steps_per_task = self.train_max_steps

    #     if not all([self.nb_tasks, self.train_max_steps, self.steps_per_task]):
    #         raise RuntimeError(
    #             "You need to provide at least two of 'max_steps', "
    #             "'nb_tasks', or 'steps_per_task'."
    #         )

    #     assert self.nb_tasks == self.train_max_steps // self.steps_per_task, (
    #         self.nb_tasks,
    #         self.train_max_steps,
    #         self.steps_per_task,
    #     )

    #     if self.test_task_schedule:
    #         if 0 not in self.test_task_schedule:
    #             raise RuntimeError("Task schedules needs to include an initial task.")

    #         if self.test_steps_per_task is not None:
    #             # If steps per task was passed, then we overwrite the number of steps
    #             # for each task in the schedule to match.
    #             self.test_task_schedule = {
    #                 i * self.test_steps_per_task: self.test_task_schedule[step]
    #                 for i, step in enumerate(sorted(self.test_task_schedule.keys()))
    #             }

    #         change_steps = sorted(self.test_task_schedule.keys())
    #         assert 0 in change_steps, "Schedule needs to include task at step 0."

    #         nb_test_tasks = len(change_steps)
    #         if self.smooth_task_boundaries:
    #             nb_test_tasks -= 1
    #         assert (
    #             nb_test_tasks == self.nb_tasks
    #         ), "nb of tasks should be the same for train and test."

    #         self.test_steps_per_task = change_steps[1] - change_steps[0]
    #         for i in range(self.nb_tasks - 1):
    #             if change_steps[i + 1] - change_steps[i] != self.test_steps_per_task:
    #                 raise NotImplementedError(
    #                     "WIP: This might not work yet if the test tasks aren't "
    #                     "equally spaced out at a fixed interval."
    #                 )

    #         self.test_steps = max(change_steps)
    #         if not self.smooth_task_boundaries:
    #             # See above note about the last entry.
    #             self.test_steps += self.test_steps_per_task

    #     elif self.test_steps_per_task is None:
    #         # This is basically never the case, since the test_steps defaults to 10_000.
    #         assert (
    #             self.test_steps
    #         ), "need to set one of test_steps or test_steps_per_task"
    #         self.test_steps_per_task = self.test_steps // self.nb_tasks
    #     else:
    #         # FIXME: This is too complicated for what is is.
    #         # Check that the test steps must either be the default value, or the right
    #         # value to use in this case.
    #         assert self.test_steps in {10_000, self.test_steps_per_task * self.nb_tasks}
    #         assert (
    #             self.test_steps_per_task
    #         ), "need to set one of test_steps or test_steps_per_task"
    #         self.test_steps = self.test_steps_per_task * self.nb_tasks

    #     assert self.test_steps // self.test_steps_per_task == self.nb_tasks

    # def create_train_wrappers(self) -> List[Callable[[gym.Env], gym.Env]]:
    #     """Get the list of wrappers to add to each training environment.

    #     The result of this method must be pickleable when using
    #     multiprocessing.

    #     Returns
    #     -------
    #     List[Callable[[gym.Env], gym.Env]]
    #         [description]
    #     """
    #     # We add a restriction to prevent users from getting data from
    #     # previous or future tasks.
    #     # TODO: Instead, just pass a subset of the task schedule to the CL wrapper?
    #     # TODO: This assumes that tasks all have the same length.
    #     # starting_step = self.current_task_id * self.steps_per_task
    #     # TODO: Ambiguous wether this `max_steps` is the maximum step that can be
    #     # reached or the maximum number of steps that can be performed.
    #     # max_steps = starting_step + self.steps_per_task - 1
    #     return self._make_wrappers(
    #         base_env=self.dataset,
    #         task_schedule=self.train_task_schedule,
    #         # TODO: Removing this, but we have to check that it doesn't change when/how
    #         # the task boundaries are given to the Method.
    #         # sharp_task_boundaries=self.known_task_boundaries_at_train_time,
    #         task_labels_available=self.task_labels_at_train_time,
    #         transforms=self.train_transforms,
    #         starting_step=starting_step,
    #         max_steps=max_steps,
    #         new_random_task_on_reset=self.stationary_context,
    #     )

    # def create_valid_wrappers(self) -> List[Callable[[gym.Env], gym.Env]]:
    #     """Get the list of wrappers to add to each validation environment.

    #     The result of this method must be pickleable when using
    #     multiprocessing.

    #     Returns
    #     -------
    #     List[Callable[[gym.Env], gym.Env]]
    #         [description]

    #     TODO: Decide how this 'validation' environment should behave in
    #     comparison with the train and test environments.
    #     """
    #     # We add a restriction to prevent users from getting data from
    #     # previous or future tasks.
    #     # TODO: Should the validation environment only be for the current task?
    #     starting_step = self.current_task_id * self.steps_per_task
    #     max_steps = starting_step + self.steps_per_task - 1
    #     return self._make_wrappers(
    #         base_env=self.val_dataset,
    #         task_schedule=self.valid_task_schedule,
    #         # sharp_task_boundaries=self.known_task_boundaries_at_train_time,
    #         task_labels_available=self.task_labels_at_train_time,
    #         transforms=self.val_transforms,
    #         starting_step=starting_step,
    #         max_steps=max_steps,
    #         new_random_task_on_reset=self.stationary_context,
    #     )
