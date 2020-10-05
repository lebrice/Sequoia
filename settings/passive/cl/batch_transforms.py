from dataclasses import dataclass, replace
from simple_parsing import list_field
from typing import Callable, Tuple, List, Union
import torch
from torch import Tensor
from settings import Observations

@dataclass
class RelabelTransform(Callable[[Tuple[Tensor, ...]], Tuple[Tensor, ...]]):
    """ Transform that puts labels back into the [0, n_classes_per_task] range.
    
    For instance, if it's given a bunch of images that have labels [2, 3, 2]
    and the `task_classes = [2, 3]`, then the new labels will be
    `[0, 1, 0]`.
    
    Note that the order in `task_classes` is perserved. For instance, in the
    above example, if `task_classes = [3, 2]`, then the new labels would be
    `[1, 0, 1]`.
    """
    task_classes: List[int] = list_field()
    
    def __call__(self, batch: Tuple[Tensor, ...]):
        if isinstance(batch, list):
            batch = tuple(batch)
        if not isinstance(batch, tuple):
            return batch
        if len(batch) == 1:
            return batch        
        x, y, *task_labels = batch
        
        if y.max() < len(self.task_classes):
            # No need to relabel this batch.
            # @lebrice: Can we really skip relabeling in this case?
            return batch

        new_y = torch.empty_like(y)
        for i, label in enumerate(self.task_classes):
            new_y[y == label] = i
        return (x, new_y, *task_labels)


@dataclass
class ReorderTensors(Callable[[Tuple[Tensor, ...]], Tuple[Tensor, ...]]):
    # reorder tensors in the batch so the task labels go into the observations:
    # (x, y, t) -> (x, t, y)
    def __call__(self, batch: Tuple[Tensor, ...]):
        if isinstance(batch, list):
            batch = tuple(batch)
        # if not isinstance(batch, tuple):
        #     return batch
        x, y, *extra_labels = batch
        if len(extra_labels) == 1:
            task_labels = extra_labels[0]
            return (x, task_labels, y)
        return batch


@dataclass
class DropTaskLabels(Callable[[Tuple[Tensor, ...]], Tuple[Tensor, ...]]):
    def __call__(self, batch: Union[Tuple[Tensor, ...], Observations]):
        if isinstance(batch, Observations):
            return replace(batch, task_labels=None)
        if not isinstance(batch, (tuple, list)):
            return batch
        if len(batch) == 3:
            return batch[0], batch[1]
        return batch
