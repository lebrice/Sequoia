""" Wrappers specific to the RL settings, so not exactly as general as those in
`common/gym_wrappers`.
"""
from collections.abc import Mapping
from dataclasses import is_dataclass, replace
from functools import singledispatch
from typing import (Any, Callable, Dict, List, Optional, Sequence, Tuple, Type,
                    TypeVar, Union)

import gym
import numpy as np
from gym import Space, spaces
from torch import Tensor

from sequoia.common import Batch
from sequoia.common.gym_wrappers import IterableWrapper, TransformObservation
from sequoia.common.spaces import Sparse
from sequoia.common.spaces.named_tuple import NamedTuple, NamedTupleSpace
from sequoia.settings.base.environment import Environment
from sequoia.settings.base.objects import (Actions, ActionType, Observations,
                                           ObservationType, Rewards,
                                           RewardType)

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
    TaskIncrementalSetting (which inherits from ClassIncrementalSetting), then
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
        
    def step(self, action: ActionType) -> Tuple[ObservationType,
                                                RewardType,
                                                Union[bool, Sequence[bool]],
                                                Union[Dict, Sequence[Dict]]]:
        # "unwrap" the actions before passing it to the wrapped environment.
        if isinstance(action, Actions):
            action = unwrap(action)
        
        observation, reward, done, info = self.env.step(action)
        # TODO: Make the observation space a Dict
        observation = self.Observations(*observation)

        reward = self.Rewards(reward)
        return observation, reward, done, info

    def reset(self, **kwargs) -> ObservationType:
        observation = self.env.reset(**kwargs)
        # TODO: Make the observation space a Dict rather than this annoying
        # NamedTuple!
        return self.Observations(*observation)


# TODO: turn unwrap into a single-dispatch callable.
# TODO: Atm 'unwrap' basically means "get rid of everything apart from the first
# item", which is a bit ugly.

@singledispatch
def unwrap(obj: Any) -> Any:
    raise NotImplementedError(obj)


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


@singledispatch
def remove_task_labels(observation: Any) -> Any:
    """ Removes the task labels from an observation / observation space. """
    if is_dataclass(observation):
        return replace(observation, task_labels=None)
    raise NotImplementedError(f"No handler registered for value {observation} of type {type(observation)}")

@remove_task_labels.register(spaces.Tuple)
@remove_task_labels.register(tuple)
def _(observation: Tuple[T, Any]) -> Tuple[T]:
    if len(observation) == 2:
        return observation[1]
    if len(observation) == 1:
        return observation[0]
    raise NotImplementedError(observation)


@remove_task_labels.register
def _remove_task_labels_in_namedtuple_space(observation: NamedTupleSpace) -> NamedTupleSpace:
    spaces = observation._spaces.copy()
    spaces.pop("task_labels")
    return type(observation)(**spaces)


@remove_task_labels.register(spaces.Dict)
@remove_task_labels.register(Mapping)
def _(observation: Dict) -> Dict:
    assert "task_labels" in observation.keys()
    return type(observation)(**{
        key: value for key, value in observation.items() if key != "task_labels"
    })



class RemoveTaskLabelsWrapper(TransformObservation):
    """ Removes the task labels from the observations and the observation space.
    """
    def __init__(self, env: gym.Env, f=remove_task_labels):
        super().__init__(env, f=f)
        self.observation_space = remove_task_labels(self.env.observation_space)

    @classmethod
    def space_change(cls, input_space: gym.Space) -> gym.Space:
        assert isinstance(input_space, spaces.Tuple), input_space
        # assert len(input_space) == 2, input_space
        return input_space[0]

from dataclasses import replace
from functools import singledispatch

from gym import Space, spaces


@singledispatch
def hide_task_labels(observation: Tuple[T, int]) -> Tuple[T, Optional[int]]:
    assert len(observation) == 2
    return observation[0], None


@hide_task_labels.register
def _hide_task_labels_on_batch(observation: Batch) -> Batch:
    return replace(observation, task_labels=None)


@hide_task_labels.register(Space)
def hide_task_labels_in_space(observation: Space) -> Space:
    raise NotImplementedError(
        f"TODO: Don't know how to remove task labels from space {observation}."
    )


@hide_task_labels.register
def _hide_task_labels_in_namedtuple_space(observation: NamedTupleSpace) -> NamedTupleSpace:
    spaces = observation._spaces.copy()
    task_label_space = spaces["task_labels"]

    if isinstance(task_label_space, Sparse):
        if task_label_space.sparsity == 1.0:
            # No need to change anything:
            return observation
        # Replace the existing 'Sparse' space with another one with the same
        # base but with sparsity = 1.0
        task_label_space = task_label_space.base

    assert not isinstance(task_label_space, Sparse)
    task_label_space = Sparse(task_label_space, sparsity=1.)
    spaces["task_labels"] = task_label_space
    return type(observation)(**spaces)


@hide_task_labels.register
def _hide_task_labels_in_tuple_space(observation: spaces.Tuple) -> spaces.Tuple:
    assert len(observation.spaces) == 2, "ambiguous"
    
    task_label_space = observation.spaces[1]
    if isinstance(task_label_space, Sparse):
        # Replace the existing 'Sparse' space with another one with the same
        # base but with sparsity = 1.0
        task_label_space = task_label_space.base
    assert not isinstance(task_label_space, Sparse)
    # We set the task label space as sparse, instead of removing that space.
    return spaces.Tuple([
        observation[0],
        Sparse(task_label_space, sparsity=1.)
    ])


@hide_task_labels.register
def hide_task_labels_in_dict_space(observation: spaces.Dict) -> spaces.Dict:
    task_label_space = observation.spaces["task_labels"]
    if isinstance(task_label_space, Sparse):
        # Replace the existing 'Sparse' space with another one with the same
        # base but with sparsity = 1.0
        task_label_space = task_label_space.base
    assert not isinstance(task_label_space, Sparse)
    return type(observation)({
        key: subspace if key != "task_labels" else Sparse(task_label_space, 1.0)
        for key, subspace in observation.spaces.items()
    })


class HideTaskLabelsWrapper(TransformObservation):
    """ Hides the task labels by setting them to None, rather than removing them
    entirely.
    
    This might be useful in order not to break the inheritance 'contract' when
    going from contexts where you don't have the task labels to contexts where
    you do have them.
    """
    def __init__(self, env: gym.Env, f=hide_task_labels):
        super().__init__(env, f=f)
        self.observation_space = hide_task_labels(self.env.observation_space)

