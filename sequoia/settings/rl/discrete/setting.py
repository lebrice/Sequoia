from dataclasses import dataclass
from ..continual.setting import ContinualRLSetting, ContinualRLTestEnvironment
from sequoia.settings.assumptions.context_discreteness import DiscreteContextAssumption
from sequoia.utils.utils import dict_union
from sequoia.settings.rl.envs import MUJOCO_INSTALLED
from typing import ClassVar, Dict, List, Callable, Optional
import gym
from simple_parsing import field


@dataclass
class DiscreteTaskAgnosticRLSetting(DiscreteContextAssumption, ContinualRLSetting):
    """ Continual Reinforcement Learning Setting where there are clear task boundaries,
    but where the task information isn't available.
    """

    # Class variable that holds the dict of available environments.
    available_datasets: ClassVar[Dict[str, str]] = dict_union(
        ContinualRLSetting.available_datasets,
        {"monsterkong": "MetaMonsterKong-v0"},
        (
            # TODO: Also add the mujoco environments for the changing sizes and masses,
            # which can't be changed on-the-fly atm.
            {}
            if not MUJOCO_INSTALLED
            else {
                # "incremental_half_cheetah": IncrementalHalfCheetahEnv
            }
        ),
    )
    
    # The number of "tasks", which in the case of Continual settings means the number of
    # base tasks that are interpolated in order to create the training, valid and test
    # nonstationarities.
    # By default 1 for this setting, meaning that the context is a linear interpolation
    # between the start context (usually the default task for the environment) and a
    # randomly sampled task.
    nb_tasks: int = field(1, alias=["n_tasks", "num_tasks"])

    # Number of steps per task. When left unset and when `max_steps` is set,
    # takes the value of `max_steps` divided by `nb_tasks`.
    steps_per_task: Optional[int] = None
    # (WIP): Number of episodes per task.
    episodes_per_task: Optional[int] = None
    # Number of steps per task in the test loop. When left unset and when `test_steps`
    # is set, takes the value of `test_steps` divided by `nb_tasks`.
    test_steps_per_task: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()
        
        assert not self.smooth_task_boundaries
         # TODO: Clean this up, not super clear what options take precedence on
        # which other options.
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

            if self.nb_tasks not in {0, 1}:
                if self.nb_tasks != nb_tasks:
                    raise RuntimeError(
                        f"Passed number of tasks {self.nb_tasks} doesn't match the "
                        f"number of tasks deduced from the task schedule ({nb_tasks})"
                    )
            self.nb_tasks = nb_tasks
            # TODO: Sort out this mess:
            if self.train_max_steps != self.nb_tasks * self.steps_per_task:
                self.train_max_steps = max(self.train_task_schedule.keys())

            # See above note about the last entry.
            self.train_max_steps += self.steps_per_task

        elif self.nb_tasks:
            if self.steps_per_task:
                self.train_max_steps = self.nb_tasks * self.steps_per_task
            elif self.train_max_steps:
                self.steps_per_task = self.train_max_steps // self.nb_tasks

        elif self.steps_per_task:
            if self.nb_tasks:
                self.train_max_steps = self.nb_tasks * self.steps_per_task
            elif self.train_max_steps:
                self.nb_tasks = self.train_max_steps // self.steps_per_task

        elif self.train_max_steps:
            if self.nb_tasks:
                self.steps_per_task = self.train_max_steps // self.nb_tasks
            elif self.steps_per_task:
                self.nb_tasks = self.train_max_steps // self.steps_per_task
            else:
                self.nb_tasks = 1
                self.steps_per_task = self.train_max_steps

        if not all([self.nb_tasks, self.train_max_steps, self.steps_per_task]):
            raise RuntimeError(
                "You need to provide at least two of 'max_steps', "
                "'nb_tasks', or 'steps_per_task'."
            )

        assert self.nb_tasks == self.train_max_steps // self.steps_per_task, (
            self.nb_tasks,
            self.train_max_steps,
            self.steps_per_task,
        )

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
            # self.steps_per_task = self.train_max_steps

        # Task schedules for training / validation and testing.

        # Create a temporary environment so we can extract the spaces and create
        # the task schedules.
        with self._make_env(
            self.dataset, self._temp_wrappers(), **self.base_env_kwargs
        ) as temp_env:
            self._setup_fields_using_temp_env(temp_env)
        del temp_env

    

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
            new_random_task_on_reset=self.stationary_context,
        )
    
    


import math
from sequoia.common.gym_wrappers import IterableWrapper
from sequoia.settings.assumptions.incremental import TaskSequenceResults
from sequoia.common.metrics.rl_metrics import EpisodeMetrics


class TestEnvironment(ContinualRLTestEnvironment, IterableWrapper):
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



