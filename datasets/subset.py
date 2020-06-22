from contextlib import contextmanager
from functools import singledispatch
from typing import (Generic, Iterable, List, Sequence, Set, Tuple, TypeVar,
                    Union)

import numpy as np
import torch
from PIL import Image as image
from PIL.Image import Image
from torch import Tensor
from torch.utils.data import ConcatDataset, Dataset
from torch.utils.data import Subset as SubsetBase
from torch.utils.data import TensorDataset
from torchvision.datasets import MNIST, ImageFolder, ImageNet, VisionDataset
from torchvision.transforms import Normalize

from common.task import Task

# from .data_utils import keep_in_memory

D = TypeVar("D", bound=Dataset)

class Subset(SubsetBase, Generic[D]):
    def __init__(self, dataset: D, indices: Sequence[int]):
        super().__init__(dataset, indices)
        self._indices: Tensor = torch.as_tensor(self.indices)

    @property
    def targets(self) -> Tensor:
        # We try to get the 'targets' attribute of the dataset.
        # If it doesn't have one, this will throw an AttributeError.
        # (This is what we want, so we put a "#type: ignore" comment here.)
        return self.dataset.targets[self._indices]  # type: ignore


class ClassSubset(Subset[D]):
    """
    Subset of a dataset containing only the given labels.
    """
    def __init__(self, dataset: D, labels: Union[Sequence[int], Task]):
        """Creates a Dataset from the x's in `dataset` whose y's are in `labels`.
        
        Args:
            dataset (Dataset): The whole Dataset.
            labels (Sequence[int]): The set of labels (targets) to keep.
        """
        self.dataset = dataset
        if isinstance(labels, Task):
            labels = labels.classes
        self.labels: Set[int] = set(labels)

        # get the mask to select only the relevant items.
        mask = get_mask(self.dataset, self.labels)
        indices = mask.nonzero().flatten()
        super().__init__(self.dataset, indices)

    def __add__(self, other: "ClassSubset") -> Union["ClassSubset", ConcatDataset]:  # type: ignore
        if isinstance(other, ClassSubset) and self.dataset is other.dataset:
            # Adding different subsets of the same dataset.
            labels = list(set(self.labels).union(set(other.labels)))
            return ClassSubset(self.dataset, labels=labels)
        return super().__add__(other)

    def __str__(self) -> str:
        return f"VisionDatasetSubset of dataset of type {type(self.dataset).__name__} with labels {self.labels}."


def get_mask(dataset: Union[VisionDataset, Subset[VisionDataset]], labels: Set[int]) -> Tensor:
    """Returns a binary mask to select only the entries with a label within `labels` from `dataset`.

    To get the corresponding indices, apply `.nonzero()` to the result.

    Parameters
    ----------
    - dataset : Union[VisionDataset, Subset[VisionDataset]]
    
        A dataset, with a `targets` attribute containing the labels.
    - labels : Sequence[int]
    
        The labels (classes) to keep.
    
    Returns
    -------
    Tensor
        A boolean mask to select the values from the dataset.
    """
    selected_mask = torch.zeros(len(dataset), dtype=torch.bool)
    targets = dataset.targets
    targets = torch.as_tensor(targets)
    for label in labels:
        selected_mask |= (targets == label)
    return selected_mask
