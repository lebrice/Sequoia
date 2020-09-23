from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Callable, ClassVar, Dict, List, Tuple, Union

import gym
import numpy as np
from gym.wrappers import TransformObservation

from common.gym_wrappers import (MultiTaskEnvironment, PixelStateWrapper,
                                 SmoothTransitions)
from common.transforms import Transforms
from settings.active.rl import GymDataLoader
from settings.active.setting import ActiveSetting
from simple_parsing import choice, list_field
from utils import dict_union
from utils.logging_utils import get_logger

logger = get_logger(__file__)


def copy_if_negative_strides(image: np.ndarray):
    if any(s < 0 for s in image.strides):
        return image.copy()
    return image


@dataclass
class ContinualRLSetting(ActiveSetting):
    available_datasets: ClassVar[Dict[str, str]] = {
        "cartpole": "CartPole-v0"
    }
    dataset: str = choice(available_datasets, default="cartpole")
    observe_state_directly: bool = False
    
    # Max number of steps ("length" of the training and test "datasets").
    max_steps: int = 1_000_000
    # Number of steps per task.
    steps_per_task: int = 100_000
    
    # Wether the task boundaries are smooth or sudden.
    smooth_task_boundaries: bool = True

    # Transforms used for all of train/val/test.
    # We use the channels_first transform when viewing the state as pixels.
    # BUG: @lebrice Added this image copy because I'm getting some weird bugs
    # because of negative strides. 
    transforms: List[Transforms] = list_field(copy_if_negative_strides, Transforms.to_tensor, Transforms.channels_first_if_needed)

    def __post_init__(self,
                      obs_shape: Tuple[int, ...] = (),
                      action_shape: Tuple[int, ...] = (),
                      reward_shape: Tuple[int, ...] = ()):
        self.task_schedule: Dict[int, Dict[str, float]] = OrderedDict()
        self.train_task_schedule: Dict[int, Dict[str, float]] = OrderedDict()
        self.val_task_schedule: Dict[int, Dict[str, float]] = OrderedDict()
        self.test_task_schedule: Dict[int, Dict[str, float]] = OrderedDict()
        
        super().__post_init__(
            # obs_shape=obs_shape,
            # action_shape=action_shape,
            # reward_shape=reward_shape,
        )
        # Create a temporary environment just to get the shapes and such.
        # TODO: Do we really need to do this though?
        # TODO: Rework all of this here.
        temp_env = self.create_temp_env()
        temp_env.reset()

        self.task_schedule = self.create_task_schedule(temp_env)

        self.observation_space = temp_env.observation_space
        self.action_space = temp_env.action_space
        self.reward_range = temp_env.reward_range

        self.obs_shape = obs_shape or self.observation_space.shape
        self.action_shape = self.action_space.shape or (1,)
        self.reward_shape = reward_shape or (1,)

        # NOTE: Here we could use a different task schedule for testing, if we
        # wanted to! However for now we will use the same tasks for training and
        # for testing:
        self.train_task_schedule = deepcopy(self.task_schedule)
        self.val_task_schedule = deepcopy(self.task_schedule)
        self.test_task_schedule = deepcopy(self.task_schedule)
        # self.test_task_schedule = self.create_task_schedule()
        
        self.train_env: GymDataLoader
        self.val_env: GymDataLoader
        self.test_env: GymDataLoader
        # close the temporary environment, we don't need it anymore.
        temp_env.close()

    def create_task_schedule(self, env: MultiTaskEnvironment = None) -> Dict[int, Dict[str, float]]:
        """Create a task schedule for the given environment.

        When `env` is not given, creates a temporary environment. We basically
        just want to get access to the `random_task()` method of the
        `MultiTaskEnvironment` wrapper for the chosen environment.

        Args:
            env (MultiTaskEnvironment, optional): The environment whose
            `random_task()` method will be used to create a task schedule.
            Defaults to None, in which case we construct a temporary
            environment.

        Returns:
            Dict[int, Dict[str, Any]: A task schedule (a dict mapping from
            step to attributes to be set on the wrapped environment).
        """
        temp_env = env or self.create_temp_env()
        task_schedule: Dict[int, Dict[str, float]] = OrderedDict()
        for step in range(0, self.max_steps, self.steps_per_task):
            task = temp_env.random_task()
            logger.debug(f"Task at step={step}: {task}")
            task_schedule[step] = task
        return task_schedule
    
    def create_temp_env(self):
        env = gym.make(self.env_name)
        for wrapper in self.env_wrappers():
            env = wrapper(env)
        return env
    
    @property
    def env_name(self) -> str:
        """Formatted name of the dataset/environment to be passed to `gym.make`.
        """
        if self.dataset in self.available_datasets:
            return self.available_datasets[self.dataset]
        elif self.dataset in self.available_datasets.values():
            return self.dataset
        else:
            logger.warning(UserWarning(
                f"dataset {self.dataset} isn't supported atm! This will try to "
                f"use it nonetheless, but you do this at your own risk!"
            ))
            return self.dataset

    def env_wrappers(self) -> List[Union[Callable, Tuple[Callable, Dict]]]:
        wrappers = []
        if not self.observe_state_directly:
            wrappers.append(PixelStateWrapper)
        if self.transforms:
            wrappers.append(partial(TransformObservation, f=self.transforms))
        if self.smooth_task_boundaries:
            wrappers.append(partial(SmoothTransitions, task_schedule=self.task_schedule))
        else:
            wrappers.append(partial(MultiTaskEnvironment, task_schedule=self.task_schedule))
        return wrappers

    # TODO: Could overwrite those to use different wrappers for train/val/test.
    def train_env_wrappers(self)-> List[Union[Callable, Tuple[Callable, Dict]]]:
        return self.env_wrappers()
    def val_env_wrappers(self) -> List[Union[Callable, Tuple[Callable, Dict]]]:
        return self.env_wrappers()
    def test_env_wrappers(self) -> List[Union[Callable, Tuple[Callable, Dict]]]:
        return self.env_wrappers()

    def train_dataloader(self, *args, **kwargs) -> GymDataLoader:
        kwargs = dict_union(self.dataloader_kwargs, kwargs)
        wrappers = self.train_env_wrappers()
        self.train_env = GymDataLoader(
            env=self.env_name,
            pre_batch_wrappers=wrappers,
            max_steps=self.max_steps,
            **kwargs
        )
        return self.train_env
    
    def val_dataloader(self, *args, **kwargs) -> GymDataLoader:
        kwargs = dict_union(self.dataloader_kwargs, kwargs)
        wrappers = self.val_env_wrappers()
        self.val_env = GymDataLoader(
            env=self.env_name,
            pre_batch_wrappers=wrappers,
            max_steps=self.max_steps,
            **kwargs
        )
        return self.val_env

    def test_dataloader(self, *args, **kwargs) -> GymDataLoader:
        kwargs = dict_union(self.dataloader_kwargs, kwargs)
        wrappers = self.test_env_wrappers()
        self.test_env = GymDataLoader(
            env=self.env_name,
            pre_batch_wrappers=wrappers,
            max_steps=self.max_steps,
            **kwargs
        )
        return self.test_env


if __name__ == "__main__":
    ContinualRLSetting.main()
