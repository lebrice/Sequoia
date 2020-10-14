from dataclasses import dataclass, replace
from functools import partial
from typing import Callable, List, Tuple, Union

import gym
import torch
from continuum.tasks import TaskSet
from gym.wrappers import TransformObservation, TransformReward
from settings import Observations, Rewards
from simple_parsing import list_field
from torch import Tensor

from ..passive_environment import PassiveEnvironment


class RelabelWrapper(TransformReward):
    def __init__(self, env: gym.Env, task_classes: List[int]):
        self.task_classes = task_classes
        super().__init__(env=env, f=partial(relabel, task_classes=self.task_classes))


@dataclass
class RelabelTransform(Callable[[Tuple[Tensor, ...]], Tuple[Tensor, ...]]):
    """ Transform that puts labels back into the [0, n_classes_per_task] range.
    
    For instance, if it's given a bunch of images that have labels [2, 3, 2]
    and the `task_classes = [2, 3]`, then the new labels will be
    `[0, 1, 0]`.
    
    Note that the order in `task_classes` is perserved. For instance, in the
    above example, if `task_classes = [3, 2]`, then the new labels would be
    `[1, 0, 1]`.
    
    IMPORTANT: This transform needs to be applied BEFORE ReorderTensor or
    SplitBatch, because it expects the batch to be (x, y, t) order
    """
    task_classes: List[int] = list_field()
    
    def __call__(self, batch: Tuple[Tensor, ...]):
        assert isinstance(batch, (list, tuple)), batch
        if len(batch) == 2:
            observations, rewards = batch
        if len(batch) == 1:
            return batch
        x, y, *task_labels = batch
        
        # if y.max() == len(self.task_classes):
        #     # No need to relabel this batch.
        #     # @lebrice: Can we really skip relabeling in this case?
        #     return batch

        new_y = relabel(y, task_classes=self.task_classes)
        return (x, new_y, *task_labels)

def relabel(y: Tensor, task_classes: List[int]) -> Tensor:
    new_y = torch.zeros_like(y)
    for i, label in enumerate(task_classes):
        new_y[y == label] = i
    return new_y

@dataclass
class ReorderTensors(Callable[[Tuple[Tensor, ...]], Tuple[Tensor, ...]]):
    # reorder tensors in the batch so the task labels go into the observations:
    # (x, y, t) -> (x, t, y)
    # TODO: Change this to:
    # (x, y, t) -> ((x, t), y) maybe?
    def __call__(self, batch: Tuple[Tensor, ...]):
        assert isinstance(batch, (list, tuple))
        if len(batch) == 2:
            observations, rewards = batch
            if isinstance(observations, Observations) and isinstance(rewards, Rewards):
                return batch
        elif len(batch) == 3:
            x, y, *extra_labels = batch
            if len(extra_labels) == 1:
                task_labels = extra_labels[0]
                return (x, task_labels, y)
        assert False, batch

@dataclass
class DropTaskLabels(Callable[[Tuple[Tensor, ...]], Tuple[Tensor, ...]]):
    def __call__(self, batch: Union[Tuple[Tensor, ...], Observations]):
        assert isinstance(batch, (tuple, list))
        if len(batch) == 2:
            observations, rewards = batch
            if isinstance(observations, Observations) and isinstance(rewards, Rewards):
                return replace(observations, task_labels=None), rewards
        elif len(batch) == 3:
            # This is tricky. If we're placed BEFORE the 'ReorderTensors',
            # then the ordering is `x, y, t`, while if we're AFTER, the
            # ordering would then be 'x, t, y'..
            x, v1, v2 = batch
            # IDEA: For now, we assume that the 'y' is a lot more erratic than
            # the task label. Therefore, the number of unique consecutive should
            # be greater for `y` than for `t`.
            u1 = len(v1.unique_consecutive())
            u2 = len(v2.unique_consecutive())
            if u1 > u2:
                y, t = v1, v2
            elif u1 == u2:
                # hmmm wtf?
                assert False, (v1, v2, u1, u2)
            else:
                y, t = v2, v1
            return x, y, t
        assert False, f"There are no task labels to drop: {batch}"
