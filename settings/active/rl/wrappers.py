from typing import Optional, Tuple, TypeVar, Type, Union

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
    to the types of Observations and Rewards, respectively, and 
    """ 
    def __init__(self,
                 env: gym.Env,
                 observations_type: Type[Observations],
                 rewards_type: Type[Rewards],
                 actions_type: Type[Actions]):
        self.Observations = observations_type
        self.Rewards = rewards_type
        self.Actions = actions_type
        env = TransformObservation(env, f=self.Observations.from_inputs)
        env = TransformReward(env, f=self.Rewards.from_inputs)
        
        def convert_action_object_to_sample_from_action_space(action: Actions):
            if isinstance(action, Batch):
                assert len(action) == 1
                action = action[0]
            if isinstance(action, Tensor):
                action = action.cpu().numpy()
            return action

        convert_action_object_to_sample_from_action_space.space_change = lambda x: x
        
        env = TransformAction(env, f=convert_action_object_to_sample_from_action_space)
        super().__init__(env=env)



def unwrap_actions(actions: Actions) -> Union[Tensor, np.ndarray]:
    if isinstance(actions, Actions):
        return actions[0]
    return actions

def unwrap_rewards(rewards: Rewards) -> Union[Tensor, np.ndarray]:
    if isinstance(rewards, Rewards):
        assert len(rewards) != 0, (rewards, rewards.field_names)
        return rewards[0]
    return rewards

def unwrap_observations(observations: Observations) -> Union[Tensor, np.ndarray]:
    if isinstance(observations, Observations):
        # TODO: Keep the task labels? or no?
        return observations.as_tuple()
    return observations


class NoTypedObjectsWrapper(IterableWrapper):
    def __init__(self, env: gym.Env):
        env = TransformObservation(env, f=unwrap_observations)
        env = TransformAction(env, f=unwrap_actions)
        env = TransformReward(env, f=unwrap_rewards)
        super().__init__(env)


def remove_task_labels(observation: Tuple[T, int]) -> T:
    assert len(observation) == 2
    return observation[0]


class RemoveTaskLabelsWrapper(TransformObservation, IterableWrapper):
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


class HideTaskLabelsWrapper(TransformObservation, IterableWrapper):
    def __init__(self, env: gym.Env, f=hide_task_labels):
        super().__init__(env, f=f)
        self.observation_space = self.space_change(self.env.observation_space)
        
    
    @classmethod
    def space_change(cls, input_space: gym.Space) -> gym.Space:
        assert isinstance(input_space, spaces.Tuple)
        # TODO: If we create something like an OptionalSpace, we
        # would replace the second part of the tuple with it. We
        # leave it the same here for now.
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

