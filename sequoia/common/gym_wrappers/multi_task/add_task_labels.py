
from functools import singledispatch
from typing import Any, Dict, Tuple, TypeVar, Union

import gym
import numpy as np
from gym import Space, spaces
from torch import Tensor

from sequoia.common.spaces.named_tuple import NamedTuple, NamedTupleSpace

X = TypeVar("X")
T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


class ObservationsAndTaskLabels(NamedTuple):
    x: Any
    task_labels: Any




@singledispatch
def add_task_labels(observation: Any, task_labels: Any) -> Any:
    raise NotImplementedError(observation, task_labels)


@add_task_labels.register(int)
@add_task_labels.register(float)
@add_task_labels.register(Tensor)
@add_task_labels.register(np.ndarray)
def _add_task_labels_to_single_obs(observation: X, task_labels: T) -> Tuple[X, T]:
    return ObservationsAndTaskLabels(x=observation, task_labels=task_labels)


@add_task_labels.register(spaces.Space)
def _add_task_labels_to_space(observation: X, task_labels: T) -> spaces.Dict:
    return NamedTupleSpace(
        x=observation, task_labels=task_labels, dtype=ObservationsAndTaskLabels,
    )


@add_task_labels.register(NamedTupleSpace)
def _add_task_labels_to_namedtuple(
    observation: NamedTupleSpace, task_labels: gym.Space
) -> NamedTupleSpace:
    """ Adding task labels to a NamedTuple Space. """
    space_dict = dict(observation.items())
    assert "task_labels" not in space_dict, "space already has task labels!"
    space_dict["task_labels"] = task_labels
    return type(observation)(space_dict, dtype=observation.dtype)


@add_task_labels.register(spaces.Tuple)
@add_task_labels.register(tuple)
def _add_task_labels_to_tuple(observation: Tuple, task_labels: T) -> Tuple:
    """ Add task labels to a tuple or a Tuple space. """
    if isinstance(observation, ObservationsAndTaskLabels):
        return observation._replace(task_labels=task_labels)
    return type(observation)([*observation, task_labels])


@add_task_labels.register(spaces.Dict)
@add_task_labels.register(dict)
def _add_task_labels_to_dict(
    observation: Union[Dict[str, V], spaces.Dict], task_labels: T
) -> Union[Dict[str, Union[V, T]], spaces.Dict]:
    """ Add task labels to a dict or a Dict space. """
    new: Dict[str, Union[V, T]] = {key: value for key, value in observation.items()}
    assert "task_labels" not in new
    new["task_labels"] = task_labels
    return type(observation)(**new)  # type: ignore
