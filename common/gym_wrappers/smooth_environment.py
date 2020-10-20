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


## TODO (@lebrice): Really cool idea!: Create a TaskSchedule class that inherits
# from Dict and when you __getitem__ a missing key, returns an interpolation! 


class SmoothTransitions(MultiTaskEnvironment):
    """ Extends MultiTaskEnvironment to support smooth task boudaries.

    Same as `MultiTaskEnvironment`, but when in between two tasks, the
    environment will have its values set to a linear interpolation of the
    attributes from the two neighbouring tasks.
    ```
    env = gym.make("CartPole-v0")
    env = SmoothTransitions(env, task_schedule={
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
                 only_update_on_episode_end: bool = False,
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
            only_update_on_episode_end (bool, optional): When `False` (default),
                update the attributes of the environment smoothly after each
                step. When `True`, only update at the end of episodes (when
                `reset()` is called).
        """
        super().__init__(env, *args, **kwargs)
        self.only_update_on_episode_end: bool = only_update_on_episode_end

    def step(self, *args, **kwargs):
        if not self.only_update_on_episode_end:
            self.smooth_update()
        return super().step(*args, **kwargs)

    def reset(self, **kwargs):
        # TODO: test this out.
        if self.only_update_on_episode_end:
            self.smooth_update()
        return super().reset(**kwargs)

    @property
    def current_task_id(self) -> Optional[int]:
        """ Returns the 'index' of the current task within the task schedule.
        
        In this case, we return None, since there aren't clear task boundaries. 
        """
        return None

    def task_array(self, task: Dict[str, float]) -> np.ndarray:
        return np.array([
            task.get(k, self.default_task[k]) for k in self.task_params
        ])

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
            interpolated_value: float = np.interp(
                x=self.steps,
                xp=steps,
                fp=fixed_points,
            )
            current_task[attr] = interpolated_value
            # logger.debug(f"interpolated value of {attr} at step {self.step}: {interpolated_value}")
        # logger.debug(f"Updating task at step {self.step}: {current_task}")
        self.current_task = current_task

