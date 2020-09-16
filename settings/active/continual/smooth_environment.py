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

from utils.logging_utils import get_logger

from .multi_task_environment import MultiTaskEnvironment

logger = get_logger(__file__)
    

class SmoothTransitions(MultiTaskEnvironment):
    """ Extends MultiTaskEnvironment to support smooth task boudaries.

    Same as `MultiTaskEnvironment`, but when in between two tasks, the
    environment will have its values set to a linear interpolation of the
    attributes from the two neighbouring tasks.
    ```
    env = gym.make("CartPole-v0")
    env = MultiTaskEnvironment(original, task_schedule={
        10: dict(length=1.0),
        20: dict(length=2.0),
    })
    env.seed(123)
    env.reset()
    ```
    
    At step 0, the length is the default value (0.5)
    at step 1, the length is 0.5 + (1 / 10) * (1.0-0.5) = 0.55
    at step 2, the length is 0.5 + (2 / 10) * (1.0-0.5) = 0.60,
    etc.

    NOTE: This only works with float attributes at the moment.

    """
    def __init__(self,
                 env: gym.Env,
                 *args,
                 only_update_on_resets: bool = False,
                 **kwargs):
        """ Wraps the environment, allowing for smooth task transitions.

        Same as `MultiTaskEnvironment`, but when in between two tasks, the
        environment will have its values set to a linear interpolation of the
        attributes from the two neighbouring tasks.


        TODO: Should we update the task paramers only on resets? or at each
        step? Might save a little bit of compute to only do it on resets, but
        then it's not exactly as 'smooth' as we would like it to be, especially
        if a single episode can be very long!

        NOTE: Assumes that the attributes are floats for now.

        Args:
            env (gym.Env): The gym environment to wrap.
            task_schedule (Dict[int, Dict[str, float]], optional) (Same as
                `MultiTaskEnvironment`): Dict mapping from a given step
                number to the attributes to be set at that time. Interpolations
                between the two neighbouring tasks will be used between task
                transitions.
            only_update_on_resets (bool, optional): When `False` (default),
                update the attributes of the environment smoothly after each
                step. When `False`, only update at the end of episodes (when
                `reset()` is called).
        """
        super().__init__(env, *args, **kwargs)
        self.only_update_on_resets: bool = only_update_on_resets

        if 0 not in self.task_schedule:
            self.task_schedule[0] = self.default_task.copy()
        
        for step in sorted(self.task_schedule.keys()):
            task = self.task_schedule[step]
            if isinstance(task, np.ndarray):
                assert len(task) == len(self.task_params), (
                    f"There has to be a value for each of {self.task_params} "
                    f"when passing a numpy array in the task schedule."
                )
                self.task_schedule[step] = dict(zip(self.task_params, task))
        # Reorder the dict based on the keys:
        # TODO: Would this be necessary if we were using a regular dict?
        self.task_schedule = OrderedDict(sorted(self.task_schedule.items()))

    def task_array(self, task: Dict[str, float]) -> np.ndarray:
        return np.array([
            task.get(k, self.default_task[k]) for k in self.task_params
        ])

    def task_dict(self, task_array: np.ndarray) -> Dict[str, float]:
        assert len(task_array) == len(self.task_params), (
            "Lengths should match the number of task parameters."
        )
        return OrderedDict(zip(self.task_params, task_array))

    def smooth_update(self) -> None:
        """ Update the curren_task at every step, based on a smooth mix of the
        previous and the next task. Every time we reach a _step that is in the
        task schedule, we update the 'prev_task_step' and 'next_task_step'
        attributes.
        """
        current_task: Dict[str, float] = OrderedDict()
        for attr in self.task_params:
            steps: List[int] = []
            # list of the
            fixed_points: List[float] = []
            for step, task in sorted(self.task_schedule.items()):
                steps.append(step)
                fixed_points.append(task.get(attr, self.default_task[attr]))
            # logger.debug(f"{attr}: steps={steps}, fp={fixed_points}")
            interpolated_value: float = np.interp(x=self._step, xp=steps, fp=fixed_points)
            current_task[attr] = interpolated_value
            # logger.debug(f"interpolated value of {attr} at step {self._step}: {interpolated_value}")
        
        # logger.debug(f"Updating task at step {self._step}: {current_task}")
        self.current_task = current_task

    def step(self, *args, **kwargs):
        self.smooth_update()
        results = super().step(*args, **kwargs)
        # self.smooth_update()
        return results

    def reset(self, **kwargs):
        return super().reset(**kwargs)
        self.smooth_update()

    # @property
    # def task_schedule(self) -> Dict[int, Dict[str, float]]:
    #     return self._task_schedule
