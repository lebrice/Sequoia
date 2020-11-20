from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple

import gym
from pytorch_lightning import LightningModule

from .gym_dataloader import GymDataLoader
from utils import constant, dict_union
from utils.logging_utils import get_logger

from .continual_rl_setting import ContinualRLSetting, HideTaskLabelsWrapper

logger = get_logger(__file__)


@dataclass
class ClassIncrementalRLSetting(ContinualRLSetting):
    """ Continual RL setting the data is divided into 'tasks' with clear boundaries.

    By default, the task labels are given at train time, but not at test time.

    TODO: Decide how to implement the train procedure, if we give a single
    dataloader, we might need to call the agent's `on_task_switch` when we reach
    the task boundary.. Or, we could produce one dataloader per task, and then
    implement a custom `fit` procedure in the CLTrainer class, that loops over
    the tasks and calls the `on_task_switch` when needed.
    """
    # Number of tasks.
    nb_tasks: int = 10

    # Wether the task boundaries are smooth or sudden.
    smooth_task_boundaries: bool = constant(False)
    # Wether to give access to the task labels at train time.
    task_labels_at_train_time: bool = True
    # Wether to give access to the task labels at test time.
    task_labels_at_test_time: bool = False

    def __post_init__(self, *args, **kwargs):
        if self.train_task_schedule:
            self.nb_tasks = len(self.train_task_schedule)
        super().__post_init__(*args, **kwargs)
        assert not self.smooth_task_boundaries
