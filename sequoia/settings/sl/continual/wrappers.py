from sequoia.common.gym_wrappers import TransformReward, IterableWrapper
from functools import singledispatch
import gym
from torch import Tensor
from continuum import TaskSet
from typing import List, Any, Dict
import numpy as np
import torch
from functools import partial


@singledispatch
def relabel(data: Any, mapping: Dict[int, int] = None) -> Any:
    """ Relabels the given data (from a task) so they all share the same action space.
    """
    raise NotImplementedError(f"Don't know how to relabel {data} of type {type(data)}")


@relabel.register
def relabel_ndarray(y: np.ndarray, mapping: Dict[int, int]=None) -> np.ndarray:
    new_y = y.copy()
    mapping = mapping or {
        c: i for i, c in enumerate(torch.unique(y))
    }
    for old_label, new_label in mapping.items():
        new_y[y == old_label] = new_label
    return new_y


@relabel.register
def relabel_tensor(y: Tensor, mapping: Dict[int, int]=None) -> Tensor:
    new_y = y.copy()
    mapping = mapping or {
        c: i for i, c in enumerate(torch.unique(y))
    }
    for old_label, new_label in mapping.items():
        new_y[y == old_label] = new_label
    return new_y



@relabel.register
def relabel_taskset(task_set: TaskSet, mapping: Dict[int, int]=None) -> TaskSet:
    mapping = mapping or {
        c: i for i, c in enumerate(task_set.get_classes())
    }
    assert not task_set.target_trsf
    # TODO: Should use the `target_trsf` of `TaskSet` instead!
    return type(task_set)(
        x=task_set._x,
        y=task_set._y,
        t=task_set._t,
        trsf=task_set.trsf,
        target_trsf=partial(relabel, mapping=mapping),
        data_type=task_set.data_type,
    )


class SharedActionSpaceWrapper(IterableWrapper):
    # """ Wrapper that gets applied to a ContinualSLEnvironment 
    def __init__(self, env: gym.Env, task_classes: List[int]):
        self.task_classes = task_classes
        super().__init__(env=env, f=partial(relabel, task_classes=self.task_classes))

