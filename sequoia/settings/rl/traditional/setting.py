""" 'Classical' RL setting.
"""
from dataclasses import dataclass
from typing import List, Callable
import gym
from ..multi_task import MultiTaskRLSetting


@dataclass
class TraditionalRLSetting(MultiTaskRLSetting):
    """ Your usual "Classical" Reinforcement Learning setting.

    Implemented as a MultiTaskRLSetting, but with a single task.
    """
    nb_tasks: int = 1

    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)
        # Set this to True, so that we switch tasks randomly all the time.
        self._new_random_task_on_reset = True

    @property
    def phases(self) -> int:
        """The number of training 'phases', i.e. how many times `method.fit` will be
        called.

        Defaults to the number of tasks, but may be different, for instance in so-called
        Multi-Task Settings, this is set to 1.
        """
        return 1

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
        else:
            new_random_task_on_reset = True
        return self._make_wrappers(
            task_schedule=self.test_task_schedule,
            # sharp_task_boundaries=self.known_task_boundaries_at_test_time,
            task_labels_available=self.task_labels_at_test_time,
            transforms=self.test_transforms,
            starting_step=0,
            max_steps=self.max_steps,
            new_random_task_on_reset=new_random_task_on_reset,
        )
