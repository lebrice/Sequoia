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
        c: i for i, c in enumerate(np.unique(y))
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
    # if mapping:
    #     assert False, mapping
    if len(task_set.get_classes()) > 2:
        assert False, f"debugging: {task_set.get_classes()}"
    mapping = mapping or {
        c: i for i, c in enumerate(task_set.get_classes())
    }
    old_y = task_set._y
    new_y = relabel(old_y, mapping=mapping)
    assert not task_set.target_trsf
    # TODO: Two options here: Either create a new 'y' array, OR add a target_trsf that
    # does the remapping. Not sure if there's a benefit in doing one vs the other atm.
    # NOTE: Choosing to replace the `y` to make sure that the concatenated datasets keep
    # the transformed y.
    new_taskset =  replace_taskset_attributes(task_set, y=new_y)
    return new_taskset
    

from sequoia.utils.generic_functions.replace import replace


@replace.register
def replace_taskset_attributes(task_set: TaskSet, **kwargs) -> TaskSet:
    new_kwargs = dict(
        x=task_set._x,
        y=task_set._y,
        t=task_set._t,
        trsf=task_set.trsf,
        target_trsf=task_set.target_trsf,
        data_type=task_set.data_type,
        bounding_boxes=task_set.bounding_boxes,
    )
    new_kwargs.update(kwargs)
    return type(task_set)(**new_kwargs)


class SharedActionSpaceWrapper(IterableWrapper):
    # """ Wrapper that gets applied to a ContinualSLEnvironment 
    def __init__(self, env: gym.Env, task_classes: List[int]):
        self.task_classes = task_classes
        super().__init__(env=env, f=partial(relabel, task_classes=self.task_classes))

