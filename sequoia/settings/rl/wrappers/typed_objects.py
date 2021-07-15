from collections.abc import Mapping
from dataclasses import is_dataclass, replace
from functools import singledispatch
from typing import (Any, Callable, Dict, List, Optional, Sequence, Tuple, Type,
                    TypeVar, Union)

import gym
import numpy as np
from gym import Space, spaces
from sequoia.common import Batch
from sequoia.common.gym_wrappers import IterableWrapper, TransformObservation
from sequoia.common.gym_wrappers.convert_tensors import supports_tensors
from sequoia.common.spaces import Sparse, TypedDictSpace
from sequoia.common.spaces.named_tuple import NamedTuple, NamedTupleSpace
from sequoia.settings.base.environment import Environment
from sequoia.settings.base.objects import (Actions, ActionType, Observations,
                                           ObservationType, Rewards,
                                           RewardType)
from torch import Tensor

T = TypeVar("T")


class TypedObjectsWrapper(IterableWrapper, Environment[ObservationType, ActionType, RewardType]):
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
    TaskIncrementalSLSetting (which inherits from ClassIncrementalSetting), then
    the observations from that setting should be isinstances (or subclasses of)
    the Observations class that this method was designed to receive!   
    """
    def __init__(self,
                 env: gym.Env,
                 observations_type: ObservationType,
                 rewards_type: RewardType,
                 actions_type: ActionType):
        self.Observations = observations_type
        self.Rewards = rewards_type
        self.Actions = actions_type
        super().__init__(env=env)

        # TODO: Also change the action and reward spaces?
        if isinstance(self.env.observation_space, (TypedDictSpace, NamedTupleSpace)):
            # Replace the space's `dtype` with `self.Observations`.
            self.observation_space = self.env.observation_space
            self.observation_space.dtype = self.Observations

        # if isinstance(self.env.observation_space, NamedTupleSpace):
        #     self.observation_space = self.env.observation_space
        #     self.observation_space.dtype = self.Observations

    def step(self, action: ActionType) -> Tuple[ObservationType,
                                                RewardType,
                                                Union[bool, Sequence[bool]],
                                                Union[Dict, Sequence[Dict]]]:
        # "unwrap" the actions before passing it to the wrapped environment.
        action = self.action(action)        
        observation, reward, done, info = self.env.step(action)
        # TODO: Make the observation space a Dict
        observation = self.observation(observation)
        reward = self.reward(reward)
        return observation, reward, done, info

    def observation(self, observation: Any) -> ObservationType:
        if isinstance(observation, self.Observations):
            return observation
        if isinstance(observation, tuple):
            # TODO: Fix this, shouldn't get tuples like this since it's quite ambiguous.
            # assert False, observation 
            return self.Observations(*observation)
        if isinstance(observation, dict):
            try:
                return self.Observations(**observation)
            except TypeError:
                assert False, (self.Observations, observation)
        assert isinstance(observation, (Tensor, np.ndarray))
        return self.Observations(observation)

    def action(self, action: ActionType) -> Any:
        # TODO: Assert this eventually
        # assert isinstance(action, Actions), action
        if isinstance(action, Actions):
            action = action.y_pred
        if isinstance(action, Tensor) and not supports_tensors(self.env.action_space):
            action = action.detach().cpu().numpy()
        if action not in self.env.action_space:
            if isinstance(self.env.action_space, spaces.Tuple):
                action = tuple(action)
        return action

    def reward(self, reward: Any) -> RewardType:
        return self.Rewards(reward)

    def reset(self, **kwargs) -> ObservationType:
        observation = self.env.reset(**kwargs)
        return self.observation(observation)
    
    def __iter__(self):
        for batch in self.env:
            if isinstance(batch, tuple) and len(batch) == 2:
                yield self.observation(batch[0]), self.reward(batch[1])
            elif isinstance(batch, tuple) and len(batch) == 1:
                yield self.observation(batch[0])
            else:
                yield self.observation(batch)

    def send(self, action: ActionType) -> RewardType:
        action = self.action(action)
        reward = self.env.send(action)
        return self.reward(reward)


# TODO: turn unwrap into a single-dispatch callable.
# TODO: Atm 'unwrap' basically means "get rid of everything apart from the first
# item", which is a bit ugly.
# Unwrap should probably be a method on the corresponding `Batch` class, which could
# maybe accept a Space to fit into?
@singledispatch
def unwrap(obj: Any) -> Any:
    return obj
    # raise NotImplementedError(obj)


@unwrap.register(int)
@unwrap.register(float)
@unwrap.register(np.ndarray)
@unwrap.register(list)
def _unwrap_scalar(v):
    return v

@unwrap.register(Actions)
def _unwrap_actions(obj: Actions) -> Union[Tensor, np.ndarray]:
    return obj.y_pred


@unwrap.register(Rewards)
def _unwrap_rewards(obj: Rewards) -> Union[Tensor, np.ndarray]:
    return obj.y


@unwrap.register(Observations)
def _unwrap_observations(obj: Observations) -> Union[Tensor, np.ndarray]:
    # This gets rid of everything except just the image.
    # TODO: Keep the task labels? or no? For now, no.        
    return obj.x


@unwrap.register(NamedTupleSpace)
def _unwrap_space(obj: NamedTupleSpace) -> Space:
    # This gets rid of everything except just the first item in the space.
    # TODO: Keep the task labels? or no? For now, no.
    return obj[0]


@unwrap.register(TypedDictSpace)
def _unwrap_space(obj: TypedDictSpace) -> spaces.Dict:
    # This gets rid of everything except just the first item in the space.
    # TODO: Keep the task labels? or no? For now, no.
    return spaces.Dict(obj.spaces)


class NoTypedObjectsWrapper(IterableWrapper):
    """ Does the opposite of the 'TypedObjects' wrapper.
    
    Can be added on top of that wrapper to strip off the typed objects it
    returns and just returns tensors/np.ndarrays instead.

    This is used for example when applying a method from stable-baselines3, as
    they only want to get np.ndarrays as inputs.

    Parameters
    ----------
    IterableWrapper : [type]
        [description]
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = unwrap(self.env.observation_space)
    
    def step(self, action):
        if isinstance(action, Actions):
            action = unwrap(action)
        if hasattr(action, "detach"):
            action = action.detach()
        assert action in self.action_space, (action, type(action), self.action_space)        
        observation, reward, done, info = self.env.step(action)
        observation = unwrap(observation)
        reward = unwrap(reward)
        return observation, reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return unwrap(observation)

