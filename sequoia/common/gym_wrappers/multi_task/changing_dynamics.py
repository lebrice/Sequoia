""" TODO: A sort of backport of the previous version of MultiTaskEnv.

"""
import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, List, Mapping, Optional, Tuple,
                    TypeVar, Union)
from functools import partial
import gym
import numpy as np
from sequoia.utils.logging_utils import get_logger

from .multi_task_env import MultiTaskEnv

logger = get_logger(__file__)


Env = TypeVar("Env", bound=gym.Env)


@dataclass
class NonStationarity(ABC):
    """ ABC for a 'task', i.e. a 'thing' that affects the environment, restricting it
    to some subset of its state-space -ish.
    """

    @abstractmethod
    def apply_to(self, env: Env) -> Env:
        """Applies this 'task' to the given environment, returning the modified env."""


@dataclass(init=False)
class ChangeEnvAttributes(NonStationarity):
    """Nonstationarity which changes the attributes of an environment.

    This modifies the environment in-place.
    """

    attribute_dict: Dict[str, Any]

    def __init__(self, attribute_dict: Dict[str, Any] = None, **kwargs):
        super().__init__()
        self.attribute_dict = attribute_dict or dict(kwargs)

    def apply_to(self, env: Env) -> Env:
        for key, value in self.attribute_dict.items():
            setattr(env.unwrapped, key, value)
        return env


@dataclass
class ApplyFunctionToEnv(NonStationarity):
    """Nonstationarity which applies a given function to the environment.
    """

    function: Callable
    args: Tuple[Any, ...] = ()
    kwargs: Dict = field(default_factory=dict)

    def apply_to(self, env: Env) -> Env:
        return self.function(env, *self.args, **self.kwargs)


NonStationarityType = TypeVar("NonStationarityType", bound=NonStationarity)


class TaskSchedule(Dict[int, NonStationarityType]):
    pass


class StepTaskSchedule(TaskSchedule[NonStationarityType]):
    pass


class EpisodeTaskSchedule(TaskSchedule[NonStationarityType]):
    pass


class ChangingDynamics(MultiTaskEnv):
    def __init__(self, env: gym.Env, tasks: Union[TaskSchedule, Dict, List[NonStationarity]]):
        if isinstance(tasks, TaskSchedule):
            self.task_schedule = tasks
            self.tasks = list(tasks.values())
        elif isinstance(tasks, list):
            self.task_schedule = {} # TODO?
            self.tasks = tasks
        elif isinstance(tasks, dict):
            # Assume the usual kind of task schedule (from the older MultiTaskEnvironment)
            self.task_schedule = StepTaskSchedule({
                step: ChangeEnvAttributes(**task) for step, task in tasks.items()
            })
            self.tasks = list(self.task_schedule.values())
        super().__init__([env for task in self.tasks])

        self._steps: int = 0
        self._episodes: int = -1

    def step(self, action):
        if self.schedule_keys_are_steps:
            task: Optional[NonStationarity] = self.task_schedule.get(self._steps)
            if task:
                logger.info(f"Applying non-stationarity {task} at step {self._steps}.")
                self.switch_tasks(task)
        obs, reward, done, info = super().step(action)
        self._steps += 1
        return obs, reward, done, info

    def switch_tasks(self, new_task_index: Union[int, NonStationarity]):
        if isinstance(new_task_index, int):
            new_task = self.tasks[new_task_index]
        elif isinstance(new_task_index, NonStationarity):
            new_task = new_task_index
            new_task_index = self.tasks.index(new_task)
        else:
            raise RuntimeError(f"Unexpected task inde: {new_task_index}.")
        assert new_task in self.task_schedule.values()
        super().switch_tasks(new_task_index)
        new_task.apply_to(self)
    
    # def get_env(self, task_index: int) -> gym.Env:
    #     """ Gets the environment at the given task index, creating it if necessary.

    #     If the envs passed to the constructor were constructors rather than gym.Env
    #     objects, the constructor for the given env will be called.
    #     """
    #     task = self.tasks[task_index]
    #     env = task.apply_to(self.env)
    #     return env

    
    
    def reset(self):
        self._episodes += 1
        if self.schedule_keys_are_episodes:
            task: Optional[NonStationarity] = self.task_schedule.get(self._episodes)
            if task:
                logger.info(f"Applying non-stationarity {task} at episode {self._episodes}.")
                task.apply_to(self)
        return super().reset()

    @property
    def schedule_keys_are_steps(self) -> bool:
        """Returns wether the schedule is step-based (vs episode-based).

        Returns
        -------
        bool
            Wether the keys in the schedule dict are steps to transition at or episodes.
        """
        return isinstance(self.task_schedule, StepTaskSchedule)

    @property
    def schedule_keys_are_episodes(self) -> bool:
        """Returns wether the schedule is episode-based (vs step-based).

        Returns
        -------
        bool
            Wether the keys in the schedule dict are episodes to transition at or steps.
        """
        return isinstance(self.task_schedule, EpisodeTaskSchedule)


class SmoothTaskSchedule(TaskSchedule[ChangeEnvAttributes]):
    def __missing__(self, key: int) -> ChangeEnvAttributes:
        assert isinstance(key, int), "this task schedule only takes ints as keys."
        current_task: Dict[str, float] = {}

        # NOTE: __missing__ won't get called if the key is in the dict.
        assert key not in self

        step_before: Optional[int] = max(
            (step for step, task in self.items() if step < key), None
        )
        step_after: Optional[int] = min(
            (step for step, task in self.items() if step > key), None
        )

        if step_before is None:
            # Return the first task.
            assert step_after is not None
            return self[step_after]

        if step_after is None:
            # Return the last task.
            return self[step_before]

        # Return an interpolation / a mix of the tasks in the schedule.
        task_before: ChangeEnvAttributes = self[step_before]
        task_after: ChangeEnvAttributes = self[step_after]
        # Gather all the attributes that will be changed by the tasks. 
        all_keys = sorted(set(itertools.chain(task.attribute_dict.keys() for task in self.values())))
        
        most_recent_values = {}
        xp: List[int] = sorted(self.keys())
        fixed_points: List[List[float]]
        for step, task in self.items():
            most_recent_values.update(task.attribute_dict)
            assert all(k in task.attribute_dict for k in all_keys), "assuming that each task has all keys."
            fixed_points.append([most_recent_values[k] for k in all_keys])

        interpolated_values = np.interp(
            x=key, xp=xp, fp=fixed_points,
        )
        interpolated_attr_dict = dict(zip(all_keys, interpolated_values))
        return ChangeEnvAttributes(attribute_dict=interpolated_attr_dict)


class SmoothStepTaskSchedule(StepTaskSchedule[ChangeEnvAttributes], SmoothTaskSchedule):
    """ Task schedule where there are smooth changes in the environment's attributes at
    each step.
    """ 
    pass


class SmoothEpisodeTaskSchedule(EpisodeTaskSchedule[ChangeEnvAttributes], SmoothTaskSchedule):
    """ Task schedule where there are smooth changes in the environment's attributes
    after each episode.
    """ 
    pass

