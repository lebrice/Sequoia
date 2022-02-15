""" 'Classical' RL setting.
"""
from dataclasses import dataclass
from typing import Callable, List

import gym

from sequoia.utils.logging_utils import get_logger
from sequoia.utils.utils import constant

from ..task_incremental import TaskIncrementalRLSetting
from ..traditional import TraditionalRLSetting

logger = get_logger(__name__)


@dataclass
class MultiTaskRLSetting(TaskIncrementalRLSetting, TraditionalRLSetting):
    """Reinforcement Learning setting where the environment alternates between a set
    of tasks sampled uniformly.

    Implemented as a TaskIncrementalRLSetting, but where the tasks are randomly sampled
    during training.
    """

    # TODO: Move this into a new Assumption about the context non-stationarity.
    stationary_context: bool = constant(True)

    @property
    def phases(self) -> int:
        """The number of training 'phases', i.e. how many times `method.fit` will be
        called.

        Defaults to the number of tasks, but may be different, for instance in so-called
        Multi-Task Settings, this is set to 1.
        """
        return 1

    # TODO: Show how the multi-task wrapper is created here, rather than in the base class.

    def create_train_wrappers(self) -> List[Callable[[gym.Env], gym.Env]]:
        return super().create_train_wrappers()

    def create_test_wrappers(self) -> List[Callable[[gym.Env], gym.Env]]:
        """Get the list of wrappers to add to a single test environment.

        The result of this method must be pickleable when using
        multiprocessing.

        Returns
        -------
        List[Callable[[gym.Env], gym.Env]]
            [description]
        """
        if self.stationary_context:
            logger.warning(
                "The test phase will go through all tasks in sequence, rather than "
                "shuffling them! (This is to make it easier to compile the performance "
                "metrics for each task."
            )
        new_random_task_on_reset = False
        # TODO: If we're in the 'Multi-Task RL' setting, then should we maybe change
        # the task schedule, so that we give an equal number of steps per task?
        return self._make_wrappers(
            base_env=self.test_dataset,
            task_schedule=self.test_task_schedule,
            # sharp_task_boundaries=self.known_task_boundaries_at_test_time,
            task_labels_available=self.task_labels_at_test_time,
            transforms=self.test_transforms,
            starting_step=0,
            max_steps=self.test_max_steps,
            new_random_task_on_reset=new_random_task_on_reset,
        )
