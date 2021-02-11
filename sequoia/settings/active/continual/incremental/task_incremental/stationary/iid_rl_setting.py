""" 'Classical' RL setting.
"""
from dataclasses import dataclass
from sequoia.utils import constant
from typing import List, Callable
import gym
from ..task_incremental_rl_setting import TaskIncrementalRLSetting

@dataclass
class RLSetting(TaskIncrementalRLSetting):
    """ Your usual "Classical" Reinforcement Learning setting.
    
    Implemented as a TaskIncrementalRLSetting, but with a single task.
    """
    nb_tasks: int = 1

    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)
        # Set this to True, so that we switch tasks randomly all the time.
        self._new_random_task_on_reset = True
        self.nb_tasks = 1
    
    def create_test_wrappers(self) -> List[Callable[[gym.Env], gym.Env]]:
        """Get the list of wrappers to add to a single test environment.
        
        The result of this method must be pickleable when using
        multiprocessing.

        Returns
        -------
        List[Callable[[gym.Env], gym.Env]]
            [description]
        """
        if self._new_random_task_on_reset:
            # TODO: If we're in the 'Multi-Task RL' setting, then should we maybe change
            # the task schedule, so that we give an equal number of steps per task?
            new_random_task_on_reset = False
        return self._make_wrappers(
            task_schedule=self.test_task_schedule,
            sharp_task_boundaries=self.known_task_boundaries_at_test_time,
            task_labels_available=self.task_labels_at_test_time,
            transforms=self.test_transforms,
            starting_step=0,
            stopping_step=self.test_steps,
            max_steps=self.max_steps,
            new_random_task_on_reset=new_random_task_on_reset,
        )
