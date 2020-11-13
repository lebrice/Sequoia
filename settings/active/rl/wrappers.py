from typing import Dict, List, Optional, Tuple, Type, TypeVar, Union

import gym
import numpy as np
from gym import spaces
from torch import Tensor

from common import Batch, batch
from common.gym_wrappers import IterableWrapper, Sparse, TransformObservation
from common.gym_wrappers.batch_env import VectorEnv
from common.gym_wrappers.transform_wrappers import (TransformAction,
                                                    TransformObservation,
                                                    TransformReward)
from common.gym_wrappers.utils import has_wrapper
from settings.base import Environment
from settings.base.objects import (Actions, ActionType, Observations,
                                   ObservationType, Rewards, RewardType)

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
        # "unwrap" the actions before passing it to the wrapped environment.
        if isinstance(action, Actions):
            action = unwrap_actions(action)
        
        observation, reward, done, info = self.env.step(action)
        observation = self.Observations.from_inputs(observation)

        reward = self.Rewards.from_inputs(reward)
        return observation, reward, done, info

    def reset(self, **kwargs) -> Observations:
        observation = self.env.reset(**kwargs)
        return self.Observations.from_inputs(observation)


def unwrap_actions(actions: Actions) -> Union[Tensor, np.ndarray]:
    assert isinstance(actions, Actions), actions
    return actions.y_pred


def unwrap_rewards(rewards: Rewards) -> Union[Tensor, np.ndarray]:
    assert isinstance(rewards, Rewards), rewards
    return rewards.y


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



def add_done(observation, done: bool):
    if isinstance(observation, tuple):
        observation =  observation + (done,)
    elif isinstance(observation, dict):
        observation["done"] = done
    else:
        observation = (observation, done)
    return observation


class AddDoneToObservation(gym.ObservationWrapper):
    # FIXME: Need to add the 'done' vector to the observation, so we can
    # get access to the 'end of episode' signal in the shared_step, since
    # when iterating over the env like a dataloader, the yielded items only
    # have the observations, and dont have the 'done' vector. (so as to be
    # consistent with supervised learning).
    
    # TODO: Should we also add the 'final state' to the observations as well?

    def __init__(self, env: gym.Env):
        super().__init__(env)
        if isinstance(env.observation_space, spaces.Tuple):
            new_spaces = list(env.observation_space.spaces)
            new_spaces.append(spaces.Discrete(2))
            self.observation_space = spaces.Tuple(new_spaces)
        elif isinstance(env.observation_space, spaces.Dict):
            new_spaces = env.observation_space.spaces.copy()
            assert "done" not in spaces, f"space shouldn't already have a 'done' key."
            new_spaces["done"] = spaces.Discrete(2)
            self.observation_space = spaces.Dict(new_spaces)
        else:
            self.observation_space = spaces.Tuple([
                self.env.observation_space,
                spaces.Discrete(2) # boolean value. (0 or 1)
            ])

    def reset(self, **kwargs):
        observation = self.env.reset()
        done = 0
        return add_done(observation, done)
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = add_done(observation, done)
        return observation, reward, done, info 

from dataclasses import replace, is_dataclass

from common.gym_wrappers.batch_env.worker import FINAL_STATE_KEY


def add_info(observation: Observations, info: List[Dict]):
    if is_dataclass(observation):
        return replace(observation, info=info)
    assert False, observation

class AddInfoToObservation(gym.ObservationWrapper):
    # TODO: Need to add the 'info' dict to the Observation, so we can have
    # access to the final observation (which gets stored in the info dict at key
    # 'final_state'.
    
    # TODO: Should we also add the 'final state' to the observations as well?

    def __init__(self, env: gym.Env):
        assert has_wrapper(env, VectorEnv), "Should only be used on vectorized environments."
        super().__init__(env)
        info_space = spaces.Dict({
            # What sparsity should we set here though?
            FINAL_STATE_KEY: spaces.Tuple([
                Sparse(env.single_observation_space)
                for _ in range(env.batch_size)
            ])
        })
                
        if isinstance(env.observation_space, spaces.Tuple):
            new_spaces = list(env.observation_space.spaces)
            new_spaces.append(info_space)
            self.observation_space = spaces.Tuple(new_spaces)

        elif isinstance(env.observation_space, spaces.Dict):
            new_spaces = env.observation_space.spaces.copy()
            assert "info" not in spaces, f"space shouldn't already have an 'info' key."
            new_spaces[info] = info_space
            self.observation_space = spaces.Dict(new_spaces)
        else:
            self.observation_space = spaces.Tuple([
                self.env.observation_space,
                info_space,
            ])

        assert False, self.observation_space

    def reset(self, **kwargs):
        observation = self.env.reset()
        info = {}
        return add_info(observation, info)
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = add_info(observation, info)
        return observation, reward, done, info 
    