from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple

import gym
from pytorch_lightning import LightningModule

from cl_trainer import CLTrainer
from common.gym_wrappers import MultiTaskEnvironment, PixelStateWrapper
from settings.active.rl import GymDataLoader
from utils import constant, dict_union
from utils.logging_utils import get_logger

from .continual_rl_setting import ContinualRLSetting

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
    # Total number of steps (including all tasks).
    max_steps: int = 1_000_000
    # Number of steps per task. When left unset, takes the value of `max_steps`
    # divided by `nb_tasks`.
    steps_per_task: int = 0
    nb_tasks: int = 10

    # Wether the task boundaries are smooth or sudden.
    smooth_task_boundaries: bool = constant(False)
    # Wether you have access to task labels at train time.
    task_labels_at_train_time: bool = True
    # Wether you have access to task labels at test time.
    task_labels_at_test_time: bool = False

    def __post_init__(self,
                      obs_shape: Tuple[int, ...] = (),
                      action_shape: Tuple[int, ...] = (),
                      reward_shape: Tuple[int, ...] = ()):

        if not self.steps_per_task:
            self.steps_per_task = self.max_steps // self.nb_tasks
        else:
            self.nb_tasks = self.max_steps // self.steps_per_task

        super().__post_init__(
            obs_shape=obs_shape,
            action_shape=action_shape,
            reward_shape=reward_shape,
        )

        self.train_tasks: List[Dict] = [task for step, task in sorted(self.train_task_schedule.items())]
        self.val_tasks: List[Dict] = [task for step, task in sorted(self.val_task_schedule.items())]
        self.test_tasks: List[Dict] = [task for step, task in sorted(self.test_task_schedule.items())]
        self._current_task_id: int = 0

    @property
    def current_task_id(self) -> int:
        return self._current_task_id

    @current_task_id.setter
    def current_task_id(self, value: int) -> int:
        logger.info(f"Setting task id: {self._current_task_id} -> {value}")
        self._current_task_id = value

    def train_env_factory(self) -> gym.Env:
        """ TODO: Idea: if task labels aren't given, we give back a single
        environment that will change between all the tasks over all the tasks
        over time.
        If task_id is given, then we give back an env specifically for that task
        that won't change over time.
        """ 
        env = self.create_gym_env()
        if self.task_labels_at_train_time:
            env.task_schedule = {}
            env.current_task = self.train_tasks[self._current_task_id]
        else:
            env.task_schedule = self.train_task_schedule
        return env

    def val_env_factory(self) -> gym.Env:
        env = self.create_gym_env()
        if self.task_labels_at_train_time:
            env.task_schedule = {}
            env.current_task = self.valid_tasks[self._current_task_id]
        else:
            env.task_schedule = self.valid_task_schedule
        return env

    def test_env_factory(self) -> gym.Env:
        env = self.create_gym_env()
        if self.task_labels_at_test_time:
            env.task_schedule = {}
            env.current_task = self.test_tasks[self._current_task_id]
        else:
            env.task_schedule = self.test_task_schedule
        return env

    def create_gym_env(self) -> MultiTaskEnvironment:
        env = gym.make(self.env_name)
        if not self.observe_state_directly:
            env = PixelStateWrapper(env)
        return MultiTaskEnvironment(env)

    def train_dataloader(self, *args, **kwargs) -> GymDataLoader:
        """
        If we have access to the task labels, then this returns a dataloader
        for the current task, hence we only do `steps_per_task` steps on that
        dataloader.
        If we don't have access to the task labels, then this returns a
        dataloader that will go over all the tasks. We then perform enough
        steps to go through all the tasks.
        """
        dataloader = super().train_dataloader(*args, **kwargs)
        if self.task_labels_at_train_time:
            dataloader.max_steps = self.steps_per_task
        else:
            dataloader.max_steps = self.max_steps
        return dataloader

    def val_dataloader(self, *args, **kwargs) -> GymDataLoader:
        dataloader = super().val_dataloader(*args, **kwargs)
        if self.task_labels_at_train_time:
            dataloader.max_steps = self.steps_per_task
        else:
            dataloader.max_steps = self.max_steps
        return dataloader

    def test_dataloader(self, *args, **kwargs) -> GymDataLoader:
        dataloader = super().test_dataloader(*args, **kwargs)
        if self.task_labels_at_test_time:
            dataloader.max_steps = self.steps_per_task
        else:
            dataloader.max_steps = self.max_steps
        return dataloader




@CLTrainer.fit_setting.register
def fit_class_incremental_rl(self, setting: ClassIncrementalRLSetting, model: LightningModule):
    n_tasks = setting.nb_tasks
    logger.info(f"Number of tasks: {n_tasks}")
    raise NotImplementedError("TODO: Use this to customize the behaviour of Trainer.fit for this setting.")
