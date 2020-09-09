"""TODO: A Wrapper that creates smooth transitions between tasks.
Could be based on the MultiTaskEnvironment, but with a moving average update of
the task, rather than setting a brand new random task.

There could also be some kind of 'task_duration' parameter, and the model does
linear or smoothed-out transitions between them depending on the step number?
"""
import bisect
from collections import OrderedDict
from typing import Dict, List, Optional

import gym
import numpy as np

from .multi_task_environment import MultiTaskEnvironment


class SmoothTransitions(MultiTaskEnvironment):
    def __init__(self,
                 env: gym.Env,
                 *args,
                 **kwargs):
        """Wraps the environment, allowing for smooth task transitions at every
        reset.

        Args:
            env (gym.Env): [description]
            task_schedule (Dict[int, Dict[str, float]], optional): Schedule
                mapping from a given step number to the state that will be set
                at that time.
        """
        super().__init__(env, *args, **kwargs)
        self.prev_task_step: int = 0
        self.next_task_step: int = 0

        self.prev_task_array: np.ndarray = self.default_task
        self.prev_task_dict: Dict[str, float] = self.default_task_dict
        
        self.next_task_array: Optional[np.ndarray] = None
        self.next_task_dict: Dict[str, float] = {}

        self.task_dicts: Dict[int, Dict[str, float]] = OrderedDict()
        self.task_arrays: Dict[int, np.ndarray] = OrderedDict()
        # List of sorted task steps.
        self.task_steps: List[int] = sorted(self.task_schedule.keys())

        for step in self.task_steps:
            task = self.task_schedule[step]
            if isinstance(task, dict):
                task_dict = self.default_task_dict.copy()
                task_dict.update(task)
                task_array = np.array([task_dict[k] for k in self.task_params])
            elif isinstance(task, np.ndarray):
                task_array = task
                assert len(task) == len(self.task_params), (
                    f"There has to be a value for each of {self.task_params} "
                    f"when passing a numpy array!"
                )
                task_dict = dict(zip(self.task_params, task))
            else:
                raise RuntimeError(
                    f"Expected to receive dicts or numpy arrays in the task "
                    f"schedule, but got {task}."
                )
            self.task_dicts[step] = task_dict
            self.task_arrays[step] = task_array

    def smooth_update(self) -> None:
        if not self.task_schedule:
            return

        # We update the task at every step, based on a smooth mix of the
        # previous and the next task. Every time we reach a _step that is in the
        # task schedule, we update the 'prev_task_step' and 'next_task_step'
        # attributes.
        if self._step in self.task_steps:
            # We are on a task boundary!
            index = self.task_steps.index(self._step)
            if index == len(self.task_steps) - 1:
                # TODO: We're at the last task in the schedule, so we keep
                # that last task as it is forever.
                return
            self.prev_task_step = self.task_steps[index]
            self.next_task_step = self.task_steps[index + 1]
            self.prev_task_array = self.task_arrays[self.prev_task_step]
            self.next_task_array = self.task_arrays[self.next_task_step]
            self.prev_task_dict = self.task_dicts[self.prev_task_step]
            self.next_task_dict = self.task_dicts[self.next_task_step]

        dist_prev = self._step - self.prev_task_step
        dist_next = self.next_task_step - self._step

        assert dist_prev >= 0 and dist_next >= 0
        # TODO: Could be interesting to try some fancier interpolation here!
        total_dist = dist_next + dist_prev
        current_task_array = (
            (dist_prev / total_dist) * self.prev_task_array +
            (dist_next / total_dist) * self.next_task_array
        )
        # Set the current task to weighted average.
        self.update_task(current_task_array)

    def step(self, *args, **kwargs):
        self.smooth_update()
        return super().step(*args, **kwargs)

    def reset(self, new_task=True, **kwargs):
        if new_task:
            self.current_task = self.p * self.current_task + (1-self.p) * self.random_task()
            self.set_current_task(self.current_task)
        return super().reset(new_task=False, **kwargs)
