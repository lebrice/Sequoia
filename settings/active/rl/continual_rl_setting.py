from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Callable, ClassVar, Dict, List, Tuple, Union

import gym
import numpy as np
from gym import Env, Wrapper
from gym.wrappers import TransformObservation

from common.gym_wrappers import (MultiTaskEnvironment, PixelStateWrapper,
                                 SmoothTransitions)
from common.gym_wrappers.utils import has_wrapper
from common.transforms import Compose, Transforms
from settings.active.rl import GymDataLoader
from settings.active.setting import ActiveSetting
from simple_parsing import choice, list_field, mutable_field
from utils import dict_union
from utils.logging_utils import get_logger

logger = get_logger(__file__)


@dataclass
class ContinualRLSetting(ActiveSetting):
    available_datasets: ClassVar[Dict[str, str]] = {
        "cartpole": "CartPole-v0"
    }
    # Which environment to learn on.
    dataset: str = choice(available_datasets, default="cartpole")
    
    # Wether we observe the internal state (angle of joints, etc) or get a pixel
    # input instead (harder).
    observe_state_directly: bool = False
    
    # Max number of steps ("length" of the training and test "datasets").
    max_steps: int = 1_000_000
    # Number of steps per task.
    steps_per_task: int = 100_000
    
    # Wether the task boundaries are smooth or sudden.
    smooth_task_boundaries: bool = True

    # Set of default transforms. Not parsed through the command-line, since it's
    # marked as a class variable.
    default_transforms: ClassVar[List[Transforms]] = [
        Transforms.to_tensor,
        Transforms.channels_first_if_needed,
    ]

    # Transforms used for all of train/val/test.
    # We use the channels_first transform when viewing the state as pixels.
    # BUG: @lebrice Added this image copy because I'm getting some weird bugs
    # because of negative strides.
    transforms: List[Transforms] = mutable_field(Compose, default_transforms)

    def __post_init__(self,
                      obs_shape: Tuple[int, ...] = (),
                      action_shape: Tuple[int, ...] = (),
                      reward_shape: Tuple[int, ...] = ()):
        self.task_schedule: Dict[int, Dict[str, float]] = OrderedDict()
        # TODO: Test out using the `Compose` as the type annotation above. If it
        # works and still allows us to parse the transforms from command line,
        # then we wouldn't need to do this here.
        logger.debug(f"self.transforms (before compose): {self.transforms}")
        self.transforms: Compose = Compose(self.transforms)
        logger.debug(f"self.transforms (after compose): {self.transforms}")

        # TODO: There is this design problem here, where we "need" to inform
        # the parent of the shape of our observations, actions, and rewards,
        # but in order to create a temporary environment, we need access to
        # some things that are usually set in the parent (like the transforms).
        # Update: Currently side-stepping this issue, by creating a 'temp' env
        # using as little state as possible (only the env name, the )
        temp_env = ContinualRLSetting.create_temp_env(
            env_name=self.env_name,
            observe_pixels=(not self.observe_state_directly),
            image_transforms=self.transforms,
        )
        temp_env.reset()
        observation_space = temp_env.observation_space
        action_space = temp_env.action_space
        reward_range = temp_env.reward_range

        obs_shape = obs_shape or observation_space.shape
        action_shape = action_shape or action_space.shape
        # The reward is always a scalar in gym environments, as far as I can tell.
        reward_shape = reward_shape or (1,)

        logger.debug(f"Observation space: {observation_space}")
        logger.debug(f"Observation shape: {obs_shape}")
        logger.debug(f"Action space: {action_space}")
        logger.debug(f"Action shape: {action_shape}")

        super().__post_init__(
            obs_shape=obs_shape,
            action_shape=action_shape,
            reward_shape=reward_shape,
        )
        # TODO: (@lebrice) I have an idea, What if  we change the postinit args
        # to be the observation space, action space, and reward space, and
        # create those spaces, even for the supervised settings! This would make
        # the API much more consistent between RL and Supervised learning!
        self.observation_space = observation_space
        self.action_space = action_space
        self.reward_range = reward_range

        # Create a task schedule. This uses the temp env just to get the
        # properties that can be set for each task.
        self.task_schedule = self.create_task_schedule_for_env(temp_env)

        # close the temporary environment, as we're done using it.
        temp_env.close()
        # TODO: Do we also need to delete it? Would that mess up the
        # observation_space or action_space variables?
        # del temp_env

        # NOTE: Here we could use a different task schedule during testing than
        # during training, if we wanted to! However for now we will use the same
        # tasks for training, validation and for testing.
        self.train_task_schedule = deepcopy(self.task_schedule)
        self.val_task_schedule = deepcopy(self.task_schedule)
        self.test_task_schedule = deepcopy(self.task_schedule)
        
        # These will be created when the `[train/val/test]_dataloader` methods
        # get called. We add them here just for type-hinting purposes.
        self.train_env: GymDataLoader
        self.val_env: GymDataLoader
        self.test_env: GymDataLoader


    def create_task_schedule_for_env(self, env: MultiTaskEnvironment) -> Dict[int, Dict[str, float]]:
        """Create a task schedule for the given environment.

        A 'Task', in this case, consists in a dictionary mapping from attribute
        names to values to be set at a given step.

        The task schedule is then a dict, mapping from steps to the
        corresponding attributes to be set and their values.

        Args:
            env (MultiTaskEnvironment, optional): The environment whose
            `random_task()` method will be used to create a task schedule.
            Defaults to None, in which case we construct a temporary
            environment.

        Returns:
            Dict[int, Dict[str, Any]: A task schedule (a dict mapping from
            step to attributes to be set on the wrapped environment).
        """
        if not has_wrapper(env, MultiTaskEnvironment):
            # We basically just want to get access to the `random_task()` method
            # of a `MultiTaskEnvironment` wrapper for the chosen environment.
            env = MultiTaskEnvironment(env)

        task_schedule: Dict[int, Dict[str, float]] = OrderedDict()
        # TODO: Do we start off with the usual, normal task?
        for step in range(0, self.max_steps, self.steps_per_task):
            task = env.random_task()
            logger.debug(f"Task at step={step}: {task}")
            task_schedule[step] = task
        return task_schedule
    
    def create_gym_env(self):
        env = gym.make(self.env_name)
        for wrapper in self.env_wrappers():
            env = wrapper(env)
        return env

    @staticmethod
    def create_temp_env(env_name: str,
                        observe_pixels: bool,
                        image_transforms: List[Callable] = None,
                        other_wrappers: List[Callable] = None):
        """
        IDEA: To try and solve the problem above (requiring the observation
        space, action space and reward shape before super().__post_init__()),
        we could have this method be different than the create_gym_env, since
        this one would only create a minimal environment which would have the 
        bare minimum wrappers needed to determine the shapes, and so it wouldn't
        depend on as many properties being set on `self`.

        NOTE: The image transforms are only added if `observe_pixels` is True.

        NOTE: Making this a static method just to highlight the intention that
        this method should depend on as few parameters as possible. Not
        allowing the `self` argument helps for that.
        """
        env = gym.make(env_name)
        if observe_pixels:
            env = PixelStateWrapper(env)
            if image_transforms:
                env = TransformObservation(env, Compose(image_transforms))
        other_wrappers = other_wrappers or []
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
