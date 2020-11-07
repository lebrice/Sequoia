from typing import Optional, Tuple, TypeVar, Type, Union, Dict

import gym
import numpy as np
from gym import spaces

from common import Batch, batch
from common.gym_wrappers import Sparse
from common.gym_wrappers import IterableWrapper, TransformObservation
from common.gym_wrappers.transform_wrappers import (TransformAction,
                                                    TransformObservation,
                                                    TransformReward)
from settings.base import Environment
from settings.base.objects import (Actions, ActionType, Observations,
                                   ObservationType, Rewards, RewardType)
from torch import Tensor

T = TypeVar("T")


class TypedObjectsWrapper(IterableWrapper):
    """ Wrapper that converts the observations and rewards coming from the env
    to `Batch` objects.
    
    NOTE: Not super necessary atm, but this would perhaps be useful if methods
    are built and expect to have a given 'type' of observations to work with,
    then any new setting that inherits from their target setting should have
    observations that subclass/inherit from the observations of their parent, so
    as not to break compatibility.
    
    For example, if a Method targets the ClassIncrementalSetting, then it
    expects to receive "observations" of the type described by
    ClassIncrementalSetting.Observations, and if it were to be applied on a
    TaskIncrementalSetting (which inherits from ClassIncrementalSetting), then
    the observations from that setting should be isinstances (or subclasses of)
    the Observations class that this method was designed to receive!   
    """
    def __init__(self,
                 env: gym.Env,
                 observations_type: Type[Observations],
                 rewards_type: Type[Rewards],
                 actions_type: Type[Actions]):
        self.Observations = observations_type
        self.Rewards = rewards_type
        self.Actions = actions_type
        super().__init__(env=env)

    def step(self, action: Actions) -> Tuple[Observations, Rewards, bool, Dict]:
        action = unwrap_actions(action)
        if hasattr(action, "detach"):
            action = action.detach()
        observation, reward, done, info = self.env.step(action)
        observation = self.Observations.from_inputs(observation)
        reward = self.Rewards.from_inputs(reward)
        return observation, reward, done, info
    
    def reset(self, **kwargs) -> Observations:
        observation = self.env.reset(**kwargs)
        return self.Observations.from_inputs(observation)


def unwrap_actions(actions: Actions) -> Union[Tensor, np.ndarray]:
    if isinstance(actions, Actions):
        actions = actions.as_tuple()
    if len(actions) == 1:
        actions = actions[0]
    return actions

def unwrap_rewards(rewards: Rewards) -> Union[Tensor, np.ndarray]:
    if isinstance(rewards, Rewards):
        # This assumes that the actions object has only one field (which is fine for now).
        rewards = rewards.as_tuple()
    if len(rewards) == 1 or (len(rewards) == 2 and rewards[1] is None):
        rewards = rewards[0]
    assert not isinstance(rewards, Rewards)
    return rewards

def unwrap_observations(observations: Observations) -> Union[Tensor, np.ndarray]:
    if isinstance(observations, Observations):
        # TODO: Keep the task labels? or no? For now, yes.        
        observations = observations.as_tuple()
    assert not isinstance(observations, Observations)
    return observations


class NoTypedObjectsWrapper(IterableWrapper):
    """ Does the opposite of the 'TypedObjects' wrapper.
    
    Can be added on top of that wrapper to strip off the typed objects it
    returns and just returns tensors/np.ndarrays instead.

    Parameters
    ----------
    IterableWrapper : [type]
        [description]
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
    
    def step(self, action):
        action = unwrap_actions(action)
        if hasattr(action, "detach"):
            action = action.detach()
        observation, reward, done, info = self.env.step(action)
        observation = unwrap_observations(observation)
        reward = unwrap_rewards(reward)
        return observation, reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return unwrap_observations(observation)


def remove_task_labels(observation: Tuple[T, int]) -> T:
    assert len(observation) == 2
    return observation[0]


class RemoveTaskLabelsWrapper(TransformObservation):
    """ Removes the task labels from the observations and the observation space.
    """
    def __init__(self, env: gym.Env, f=remove_task_labels):
        super().__init__(env, f=f)
        self.observation_space = self.space_change(self.env.observation_space)

    @classmethod
    def space_change(cls, input_space: gym.Space) -> gym.Space:
        assert isinstance(input_space, spaces.Tuple), input_space
        assert len(input_space) == 2
        return input_space[0]


def hide_task_labels(observation: Tuple[T, int]) -> Tuple[T, Optional[int]]:
    assert len(observation) == 2
    if isinstance(observation, Batch):
        return type(observation).from_inputs((observation[0], None))
    return observation[0], None


class HideTaskLabelsWrapper(TransformObservation):
    """ Hides the task labels by setting them to None, rather than removing them
    entirely.
    
    This might be useful in order not to break the inheritance 'contract' when
    going from contexts where you don't have the task labels to contexts where
    you do have them.
    """
    def __init__(self, env: gym.Env, f=hide_task_labels):
        super().__init__(env, f=f)
        self.observation_space = self.space_change(self.env.observation_space)
        
    
    @classmethod
    def space_change(cls, input_space: gym.Space) -> gym.Space:
        assert isinstance(input_space, spaces.Tuple)
        assert len(input_space) == 2
        
        task_label_space = input_space.spaces[1]
        if isinstance(task_label_space, Sparse):
            # Replace the existing 'Sparse' space with another one with the same
            # base but with none_prob = 1.0
            task_label_space = task_label_space.base
        assert not isinstance(task_label_space, Sparse)
        # Do we set the task label space as sparse? or do we just remote that
        # space?
        return spaces.Tuple([
            input_space[0],
            Sparse(task_label_space, none_prob=1.)
        ])

