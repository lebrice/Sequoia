from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from typing import ClassVar, Dict, List, Tuple

import gym
from gym.wrappers.pixel_observation import PixelObservationWrapper

from common.transforms import ChannelsFirst, Transforms
from settings.active.rl import GymDataLoader
from simple_parsing import choice, list_field
from utils import dict_union
from utils.logging_utils import get_logger

from .. import ActiveSetting
from .multi_task_environment import MultiTaskEnvironment
from .smooth_environment import SmoothTransitions
from common.gym_wrappers import PixelStateWrapper
logger = get_logger(__file__)

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

    # Transforms used for all of train/val/test.
    # We use the channels_first transform when viewing the state as pixels. 
    transforms: List[Transforms] = list_field(Transforms.to_tensor, Transforms.channels_first_if_needed)

    def __post_init__(self,
                      obs_shape: Tuple[int, ...] = (),
                      action_shape: Tuple[int, ...] = (),
                      reward_shape: Tuple[int, ...] = ()):
        self.task_schedule: Dict[int, Dict[str, float]] = OrderedDict()

        # Create a temporary environment just to get the shapes and such.
        temp_env: SmoothTransitions = self.create_gym_env()

        self.observation_space = temp_env.observation_space
        self.action_space = temp_env.action_space
        self.reward_range = temp_env.reward_range

        obs_shape = obs_shape or self.observation_space.shape
        action_shape = action_shape or self.action_space.shape or (1,)
        reward_shape = reward_shape or (1,)
        super().__post_init__(
            obs_shape=obs_shape,
            action_shape=action_shape,
            reward_shape=reward_shape,
        )

        # NOTE: Here we could use a different task schedule for testing, if we
        # wanted to! However for now we will use the same tasks for training and
        # for testing:
        self.train_task_schedule = self.create_task_schedule(temp_env)
        self.valid_task_schedule = deepcopy(self.train_task_schedule)
        self.test_task_schedule = deepcopy(self.train_task_schedule)
        # self.test_task_schedule = self.create_task_schedule()
        
        self._train_loader: GymDataLoader
        self._val_loader: GymDataLoader
        self._test_loader: GymDataLoader
        # close the temporary environment, we don't need it anymore.
        temp_env.close()

    def create_task_schedule(self, env: MultiTaskEnvironment = None) -> Dict[int, Dict[str, float]]:
        """Create a task schedule for the given environment.

        When `env` is not given, creates a temporary environment. We basically
        just want to get access to the `random_task()` method of the
        `MultiTaskEnvironment` object.

        Args:
            env (MultiTaskEnvironment, optional): The environment whose
            `random_task()` method will be used to create a task schedule.
            Defaults to None, in which case we construct a temporary
            environment.

        Returns:
            Dict[int, Dict[str, Any]: A task schedule (a dict mapping from
            step to attributes to be set on the wrapped environment).
        """
        temp_env = env or self.create_gym_env()
        task_schedule: Dict[int, Dict[str, float]] = OrderedDict()
        for step in range(0, self.max_steps, self.steps_per_task):
            task = temp_env.random_task()
            logger.debug(f"Task at step={step}: {task}")
            task_schedule[step] = task
        return task_schedule

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

    def create_gym_env(self) -> gym.Env:
        env = gym.make(self.env_name)
        if not self.observe_state_directly:
            env.reset()
            env = PixelStateWrapper(env)
        return SmoothTransitions(env)

    def setup(self, stage=None):
        return super().setup(stage=stage)

    def train_env_factory(self) -> gym.Env:
        env = self.create_gym_env()
        env.task_schedule = self.train_task_schedule
        return env

    def valid_env_factory(self) -> gym.Env:
        env = self.create_gym_env()
        env.task_schedule = self.val_task_schedule
        return env

    def test_env_factory(self) -> gym.Env:
        env = self.create_gym_env()
        env.task_schedule = self.test_task_schedule
        return env

    def train_dataloader(self, *args, **kwargs) -> GymDataLoader:
        kwargs = dict_union(self.dataloader_kwargs, kwargs)
        kwargs["num_workers"] = kwargs["batch_size"]
        self._train_loader = GymDataLoader(
            env_factory=self.train_env_factory,
            max_steps=self.max_steps,
            transforms=self.train_transforms,
            **kwargs
        )
        return self._train_loader
