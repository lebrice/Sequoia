from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple

import gym

from common.gym_wrappers import Ca, MultiTaskEnvironment, PixelStateWrapper
from settings.active.rl import GymDataLoader
from utils import dict_union

from .continual_rl_setting import ContinualRLSetting


@dataclass
class ClassIncrementalRLSetting(ContinualRLSetting):
    """TODO: Figure out how to setup the 'epochs' and the tasks for RL.
    """
    # Max number of steps ("length" of the training and test "datasets").
    max_steps: int = 1_000_000
    # Number of steps per task.
    steps_per_task: int = 100_000
    nb_tasks: int = 10

    task_labels_at_train_time: bool = True
    task_labels_at_test_time: bool = False

    def __post_init__(self,
                      obs_shape: Tuple[int, ...] = (),
                      action_shape: Tuple[int, ...] = (),
                      reward_shape: Tuple[int, ...] = ()):
        super().__post_init__(
            obs_shape=obs_shape,
            action_shape=action_shape,
            reward_shape=reward_shape,
        )
        self.train_env_factories: List[Callable[[], gym.Env]] = [
            self.make_train_env_factory(task_id) for task_id in range(self.nb_tasks)
        ]
        self.val_env_factories: List[Callable[[], gym.Env]] = [
            self.make_val_env_factory(task_id) for task_id in range(self.nb_tasks)
        ]
        self.test_env_factories: List[Callable[[], gym.Env]] = [
            self.make_test_env_factory(task_id) for task_id in range(self.nb_tasks)
        ]
        self._current_task_id: int = 0

    def make_train_env_factory(self, task_id: int = None) -> gym.Env:
        def _train_env_factory():
            env = self.create_gym_env()
            if task_id is None:
                env.task_schedule = self.train_task_schedule
            else:
                env.current_task = list(self.train_task_schedule.values)[task_id]
            return env
        return _train_env_factory

    def make_val_env_factory(self, task_id: int = None) -> gym.Env:
        def _val_env_factory():
            env = self.create_gym_env()
            if task_id is None:
                env.task_schedule = self.valid_task_schedule
            else:
                env.current_task = self.valid_task_schedule.values[task_id]
            return env
        return _val_env_factory

    def make_test_env_factory(self, task_id: int = None) -> gym.Env:
        def _test_env_factory():
            env = self.create_gym_env()
            if task_id is None:
                env.task_schedule = self.test_task_schedule
            else:
                env.current_task = self.test_task_schedule.values[task_id]
            return env
        return _test_env_factory

    def create_gym_env(self) -> MultiTaskEnvironment:
        env = gym.make(self.env_name)
        if not self.observe_state_directly:
            env = PixelStateWrapper(env)
        return MultiTaskEnvironment(env)

    def train_dataloader(self, *args, **kwargs) -> GymDataLoader:
        if self.task_labels_at_train_time:
            # TODO: Let the model know when a task switched.
            pass
        kwargs = dict_union(self.dataloader_kwargs, kwargs)
        kwargs["num_workers"] = kwargs["batch_size"]
        # Return a single GymDataLoader that will go over all the tasks incrementally.    
        self._train_loader = GymDataLoader(
            env_factory=self.train_env_factories[self._current_task_id],
            max_steps=self.steps_per_task,
            transforms=self.train_transforms,
            **kwargs
        )
        return self._train_loader


    def val_dataloader(self, *args, **kwargs) -> GymDataLoader:
        if self.task_labels_at_train_time:
            # TODO: Let the model know when a task switched.
            pass
        kwargs = dict_union(self.dataloader_kwargs, kwargs)
        kwargs["num_workers"] = kwargs["batch_size"]
        # Return a single GymDataLoader that will go over all the tasks incrementally.    
        self._val_loader = GymDataLoader(
            env_factory=self.val_env_factories[self._current_task_id],
            max_steps=self.steps_per_task,
            transforms=self.val_transforms,
            **kwargs
        )
        return self._val_loader

    def test_dataloader(self, *args, **kwargs) -> GymDataLoader:
        if self.task_labels_at_test_time:
            # TODO: Let the model know when a task switch occurs.
            pass
        kwargs = dict_union(self.dataloader_kwargs, kwargs)
        kwargs["num_workers"] = kwargs["batch_size"]
        # Return a single GymDataLoader that will go over all the tasks incrementally.    
        self._test_loader = GymDataLoader(
            env_factory=self.test_env_factories[self._current_task_id],
            max_steps=self.max_steps,
            transforms=self.test_transforms,
            **kwargs
        )
        return self._test_loader

    def train_dataloaders(self, *args, **kwargs) -> Iterable[GymDataLoader]:
        if self.task_labels_at_train_time:
            # TODO: Let the model know when a task switched.
            pass
        kwargs = dict_union(self.dataloader_kwargs, kwargs)
        kwargs["num_workers"] = kwargs["batch_size"]
        for task_id in range(self.nb_tasks):
            # Return a single GymDataLoader that will go over all the tasks incrementally.    
            task_loader = GymDataLoader(
                env_factory=self.train_env_factories[task_id],
                max_steps=self.steps_per_task,
                transforms=self.train_transforms,
                **kwargs
            )
            yield task_loader

    def val_dataloaders(self, *args, **kwargs) -> Iterable[GymDataLoader]:
        if self.task_labels_at_train_time:
            # TODO: Let the model know when a task switched.
            pass
        kwargs = dict_union(self.dataloader_kwargs, kwargs)
        kwargs["num_workers"] = kwargs["batch_size"]
        for task_id in range(self.nb_tasks):
            # Return a single GymDataLoader that will go over all the tasks incrementally.    
            task_loader = GymDataLoader(
                env_factory=self.val_env_factories[task_id],
                max_steps=self.steps_per_task,
                transforms=self.val_transforms,
                **kwargs
            )
            yield task_loader

    def test_dataloaders(self, *args, **kwargs) -> Iterable[GymDataLoader]:
        if self.task_labels_at_test_time:
            # TODO: Let the model know when a task switched.
            pass
        kwargs = dict_union(self.dataloader_kwargs, kwargs)
        kwargs["num_workers"] = kwargs["batch_size"]
        for task_id in range(self.nb_tasks):
            # Return a single GymDataLoader that will go over all the tasks incrementally.    
            task_loader = GymDataLoader(
                env_factory=self.test_env_factories[task_id],
                max_steps=self.steps_per_task,
                transforms=self.test_transforms,
                **kwargs,
            )
            yield task_loader