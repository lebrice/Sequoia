from collections.abc import Mapping
from dataclasses import is_dataclass, replace
from functools import singledispatch
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import gym
import numpy as np
from gym import Space, spaces
from torch import Tensor

from sequoia.common import Batch
from sequoia.common.gym_wrappers import IterableWrapper, TransformObservation
from sequoia.common.gym_wrappers.multi_task_environment import add_task_labels
from sequoia.common.gym_wrappers.utils import IterableWrapper
from sequoia.common.spaces import Sparse, TypedDictSpace
from sequoia.common.spaces.named_tuple import NamedTuple, NamedTupleSpace
from sequoia.settings.base.environment import Environment
from sequoia.settings.base.objects import (
    Actions,
    ActionType,
    Observations,
    ObservationType,
    Rewards,
    RewardType,
)

T = TypeVar("T")


@singledispatch
def hide_task_labels(observation: Tuple[T, int]) -> Tuple[T, Optional[int]]:
    assert len(observation) == 2
    return observation[0], None


@hide_task_labels.register(dict)
def _hide_task_labels_in_dict(observation: Dict) -> Dict:
    new_observation = observation.copy()
    assert "task_labels" in observation
    new_observation["task_labels"] = None
    return new_observation


@hide_task_labels.register
def _hide_task_labels_on_batch(observation: Batch) -> Batch:
    return replace(observation, task_labels=None)


@hide_task_labels.register(Space)
def hide_task_labels_in_space(observation: Space) -> Space:
    raise NotImplementedError(
        f"TODO: Don't know how to remove task labels from space {observation}."
    )


@hide_task_labels.register
def _hide_task_labels_in_namedtuple_space(
    observation: NamedTupleSpace,
) -> NamedTupleSpace:
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
    task_label_space = Sparse(task_label_space, sparsity=1.0)
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
    return spaces.Tuple([observation[0], Sparse(task_label_space, sparsity=1.0)])


@hide_task_labels.register
def hide_task_labels_in_dict_space(observation: spaces.Dict) -> spaces.Dict:
    task_label_space = observation.spaces["task_labels"]
    if isinstance(task_label_space, Sparse):
        # Replace the existing 'Sparse' space with another one with the same
        # base but with sparsity = 1.0
        task_label_space = task_label_space.base
    assert not isinstance(task_label_space, Sparse)
    return type(observation)(
        {
            key: subspace if key != "task_labels" else Sparse(task_label_space, 1.0)
            for key, subspace in observation.spaces.items()
        }
    )


@hide_task_labels.register(TypedDictSpace)
def hide_task_labels_in_typed_dict_space(
    observation: TypedDictSpace[T],
) -> TypedDictSpace[T]:
    task_label_space = observation.spaces["task_labels"]
    if isinstance(task_label_space, Sparse):
        # Replace the existing 'Sparse' space with another one with the same
        # base but with sparsity = 1.0
        task_label_space = task_label_space.base
    assert not isinstance(task_label_space, Sparse)
    return type(observation)(
        {
            key: subspace if key != "task_labels" else Sparse(task_label_space, 1.0)
            for key, subspace in observation.spaces.items()
        },
        dtype=observation.dtype,
    )


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


@singledispatch
def remove_task_labels(observation: Any) -> Any:
    """ Removes the task labels from an observation / observation space. """
    if is_dataclass(observation):
        return replace(observation, task_labels=None)
    raise NotImplementedError(
        f"No handler registered for value {observation} of type {type(observation)}"
    )


@remove_task_labels.register(spaces.Tuple)
@remove_task_labels.register(tuple)
def _(observation: Tuple[T, Any]) -> Tuple[T]:
    if len(observation) == 2:
        return observation[1]
    if len(observation) == 1:
        return observation[0]
    raise NotImplementedError(observation)


@remove_task_labels.register
def _remove_task_labels_in_namedtuple_space(
    observation: NamedTupleSpace,
) -> NamedTupleSpace:
    spaces = observation._spaces.copy()
    spaces.pop("task_labels")
    return type(observation)(**spaces)


@remove_task_labels.register(spaces.Dict)
@remove_task_labels.register(Mapping)
def _(observation: Dict) -> Dict:
    assert "task_labels" in observation.keys()
    return type(observation)(
        **{key: value for key, value in observation.items() if key != "task_labels"}
    )


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


class FixedTaskLabelWrapper(IterableWrapper):
    """ Wrapper that adds always the same given task id to the observations.

    Used when the list of envs for each task is passed, so that each env also has the
    task id as part of their observation space and in their observations.
    """

    def __init__(
        self, env: gym.Env, task_label: Optional[int], task_label_space: gym.Space
    ):
        super().__init__(env=env)
        self.task_label = task_label
        self.task_label_space = task_label_space
        self.observation_space = add_task_labels(
            self.env.observation_space, task_labels=task_label_space
        )

    def observation(self, observation: Union[ObservationType, Any]) -> ObservationType:
        return add_task_labels(observation, self.task_label)

    def reset(self):
        return self.observation(super().reset())

    def step(self, action):
        obs, reward, done, info = super().step(action)
        return self.observation(obs), reward, done, info
