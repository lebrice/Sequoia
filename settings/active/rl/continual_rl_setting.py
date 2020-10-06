from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Callable, ClassVar, Dict, List, Sequence, Tuple, Union, Optional, Type

import gym
import numpy as np
import torch
from gym import Env, Wrapper, spaces
from simple_parsing import choice, list_field, mutable_field
from torch import Tensor

from common.gym_wrappers import (MultiTaskEnvironment, PixelStateWrapper,
                                 SmoothTransitions, TransformObservation,
                                 has_wrapper)
from common.gym_wrappers.env_dataset import EnvDatasetItem, StepResult
from common.config import Config
from common.transforms import ChannelsFirstIfNeeded, Compose, Transforms
from utils import dict_union, get_logger

from ..active_setting import ActiveSetting
from .gym_dataloader import GymDataLoader


from settings.method_abc import MethodABC
from settings.assumptions.incremental import IncrementalSetting
from settings.base import Observations, Rewards, Actions, Results
logger = get_logger(__file__)

from .rl_results import RLResults

@dataclass
class ContinualRLSetting(ActiveSetting, IncrementalSetting):
    """ Reinforcement Learning Setting where the environment changes over time.

    This is an Active setting which uses gym environments as sources of data.
    These environments' attributes could change over time following a task
    schedule. An example of this could be that the gravity increases over time
    in cartpole, making the task progressively harder as the agent interacts with
    the environment.
    """
    Results: ClassVar[Type[Results]] = RLResults
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)
    @dataclass(frozen=True)
    class Observations(IncrementalSetting.Observations,
                       ActiveSetting.Observations):
        """ Observations in an RL Setting. """
        # Just as a reminder, these are the fields defined in the base classes:
        # x: Tensor
        # task_labels: Union[Optional[Tensor], Sequence[Optional[Tensor]]] = None
        
        @classmethod
        def from_inputs(cls, inputs: Tuple[Tensor, bool, Dict]):
            """ We customize this class method for the RL setting. """
            if isinstance(inputs, (list, tuple)) and len(inputs) == 3:
                obs, done, info = inputs
                x = obs
                task_labels = None
                # TODO: Add an "Observations transform" which adds the task labels
                # to the observations.
                return cls(x=x, task_labels=task_labels)
            return super().from_inputs(inputs)

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
    transforms: List[Transforms] = list_field(ChannelsFirstIfNeeded())

    def __post_init__(self):
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
        from gym.spaces import Dict as SpaceDict
        self.observation_space = SpaceDict({
            "x": temp_env.observation_space
        })
        self.action_space = temp_env.action_space
        self.reward_space = getattr(temp_env, "reward_space", None)
        if self.reward_space is None:
            # The reward is always a scalar in gym environments, as far as I can
            # tell.
            reward_range = temp_env.reward_range
            from gym.spaces import Box
            self.reward_space = Box(low=reward_range[0], high=reward_range[1], shape=())
        
        logger.debug(f"Observation space: {self.observation_space}")
        logger.debug(f"Action space: {self.action_space}")
        logger.debug(f"Reward space: {self.reward_space}")

        super().__post_init__(
            observation_space=self.observation_space,
            action_space=self.action_space,
            reward_space=self.reward_space,
        )
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
        self.train_env: GymDataLoader = None
        self.val_env: GymDataLoader = None
        self.test_env: GymDataLoader = None

    def apply(self, method: MethodABC, config: Config):
        self.config = config
        method.config = config
        method.configure(self)
        self.configure(method)
        self.train_loop(method)
        results = self.test_loop(method)
        # method.validate_results(self, results)
        return results
  
    def train_dataloader(self, *args, **kwargs) -> GymDataLoader[Observations, Actions, Rewards]:
        kwargs = dict_union(self.dataloader_kwargs, kwargs)
        wrappers = self.train_wrappers()
        if self.train_env:
            self.train_env.close()
            del self.train_env
        self.train_env = self.make_env_dataloader(wrappers, *args, **kwargs)
        return self.train_env
    
    def val_dataloader(self, *args, **kwargs) -> GymDataLoader[Observations, Actions, Rewards]:
        kwargs = dict_union(self.dataloader_kwargs, kwargs)
        wrappers = self.val_wrappers()
        if self.val_env:
            self.val_env.close()
            del self.val_env
        self.val_env = self.make_env_dataloader(wrappers, *args, **kwargs)
        return self.val_env
        
    def test_dataloader(self, *args, **kwargs) -> GymDataLoader[Observations, Actions, Rewards]:
        kwargs = dict_union(self.dataloader_kwargs, kwargs)
        wrappers = self.test_wrappers()
        if self.test_env:
            self.test_env.close()
            del self.test_env
        self.test_env = self.make_env_dataloader(wrappers, *args, **kwargs)
        return self.test_env

    def make_observations(self, state: Union[np.ndarray, Tensor]) -> Observations:
        # WIP: Convert the 'state' part of the EnvDatasetItem into an Observation.
        assert isinstance(state, (np.ndarray, torch.Tensor))
        x = torch.as_tensor(state)
        observations = self.Observations(x)
        return observations

    def make_env_dataloader(self, wrappers, *args, **kwargs):
        # TODO: Figure this stuff out:
        on_missing_action = self.on_missing_action
        max_steps = self.max_steps
        
        env: GymDataLoader = GymDataLoader(
            env=self.env_name,
            pre_batch_wrappers=wrappers,
            max_steps=5,
            observations_type=self.Observations,
            **kwargs,
        )
        return env
        
        
        from common.gym_wrappers.env_dataset import TransformEnvDatasetItem
        env: GymDataLoader = GymDataLoader(
            env=self.env_name,
            pre_batch_wrappers=wrappers,
            # post_batch_wrappers=post_batch_wrappers,
            max_steps=max_steps,
            on_missing_action=on_missing_action,
            # observations_type=self.Observations,
            # actions_type=self.Actions,
            # rewards_type=self.Rewards,
            **kwargs
        )
        # from common.gym_wrappers.env_dataset import TransformEnvDatasetItem
        # env = TransformObservation(env, f=self.make_observations)
        
        # TODO: The state is still a Tensor or np.ndarray, not an Observations.
        # This will also be true when iterating over the env with .step().
        # state = env.reset()
        
        # thing = env.step(env.action_space.sample())
        # assert False, thing
        # BUG: When iterating over the env, it always gives back EnvDatasetItems
        # with a Tensor as the observation, instead of giving the Observations
        # we'd like.
        # for obs_batch in env:
        #     assert isinstance(obs_batch, self.Observations), obs_batch
        #     break
        
        batch_size = env.observation_space.shape[0]
        assert isinstance(env.action_space, spaces.Tuple)
        assert len(env.action_space) == batch_size
        assert self.action_space == env.action_space[0]
        
        assert isinstance(env.reward_space, spaces.Tuple)
        assert len(env.reward_space) == batch_size
        assert self.reward_space == env.reward_space[0]
        # Update the observation/action spaces on `self` to have the batch size?
        # TODO: Not sure if this is a good idea..
        self.observation_space["x"].shape = (
            batch_size,
            *self.observation_space["x"].shape
        )
        self.action_space = env.action_space
        self.reward_space = env.reward_space
        return env
    

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
        if self.smooth_task_boundaries:
            wrappers.append(partial(SmoothTransitions, task_schedule=self.task_schedule))
        else:
            wrappers.append(partial(MultiTaskEnvironment, task_schedule=self.task_schedule))
        return wrappers

    def on_missing_action(self,
                          observation: EnvDatasetItem,
                          action_space: gym.Space) -> Tensor:
        """Called whenever a GymDataloader is missing an action when iterating.
        """
        # return action_space.sample()
        raise RuntimeError(
            "You need to send an action using the `send` method "
            "every time you get a value from the dataset! "
            "Otherwise, you can also set the the `on_missing_action` method "
            "to return a 'filler' action given the current context. "
        )
        return None

    # TODO: Could overwrite those to use different wrappers for train/val/test.
    def train_wrappers(self)-> List[Union[Callable, Tuple[Callable, Dict]]]:
        wrappers = self.env_wrappers()
        if not self.observe_state_directly:
            wrappers.append(partial(TransformObservation, f=self.train_transforms))
        return wrappers

    def val_wrappers(self) -> List[Union[Callable, Tuple[Callable, Dict]]]:
        wrappers = self.env_wrappers()
        if not self.observe_state_directly:
            wrappers.append(partial(TransformObservation, f=self.test_transforms))
        return wrappers

    def test_wrappers(self) -> List[Union[Callable, Tuple[Callable, Dict]]]:
        wrappers = self.env_wrappers()
        if not self.observe_state_directly:
            wrappers.append(partial(TransformObservation, f=self.test_transforms))
        return wrappers

    def _check_dataloaders_give_correct_types(self):
        """ Do a quick check to make sure that the dataloaders give back the
        right observations / reward types.
        """
        # TODO: This method is duplicated in a few places just for debugging atm:
        # (ClassIncrementalSetting, Setting, and here).
        for loader_method in [self.train_dataloader, self.val_dataloader, self.test_dataloader]:
            env = loader_method()
            
            from settings.passive import PassiveEnvironment
            from settings.active import ActiveEnvironment
            from utils.utils import take
            
            for i, batch in zip(range(5), env):
                logger.debug(f"Checking at step {i} in env {env}")
                observations, rewards = batch, None

                if not isinstance(observations, self.Observations):
                    assert False, (type(observations), [type(v) for v in observations])

                observations: Observations
                batch_size = observations.batch_size
                batch_size = observations.batch_size
                rewards: Optional[Rewards] = rewards[0] if rewards else None
                if rewards is not None:
                    assert isinstance(rewards, self.Rewards), type(rewards)
                # TODO: If we add gym spaces to all environments, then check
                # that the observations are in the observation space, sample
                # a random action from the action space, check that it is
                # contained within that space, and then get a reward by
                # sending it to the dataloader. Check that the reward
                # received is in the reward space.
                actions = self.action_space.sample()
                assert len(actions) == batch_size
                if not isinstance(actions, self.Actions):
                    actions = self.Actions(actions)
                rewards = env.send(actions)
                assert isinstance(rewards, self.Rewards), type(rewards)
            

if __name__ == "__main__":
    ContinualRLSetting.main()
