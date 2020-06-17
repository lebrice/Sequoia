from contextlib import contextmanager
from typing import Iterable, List, Sequence, Set, Tuple, Union

import numpy as np
import torch
from PIL import Image as image
from PIL.Image import Image
from torch import Tensor
from torch.utils.data import TensorDataset, Dataset, Subset as SubsetBase, ConcatDataset
from torchvision.transforms import Normalize
from common.task import Task
from torchvision.datasets import VisionDataset, MNIST

class Subset(SubsetBase):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        if isinstance(self.dataset, VisionDataset):
            from datasets.dataset import fix_vision_dataset
            fix_vision_dataset(self.dataset)
        if not isinstance(self.indices, Tensor):
            self.indices = torch.as_tensor(self.indices)

    @property
    def data(self) -> Tensor:
        return self.dataset.data[self.indices]

    @property
    def targets(self) -> Tensor:
        return self.dataset.targets[self.indices]


class ClassSubset(TensorDataset):
    """
    Subset of a dataset containing only the given labels.
    """
    def __init__(self, dataset: Dataset, labels: Union[Sequence[int], Task]):
        """Creates a Dataset from the x's in `dataset` whose y's are in `labels`.
        
        Args:
            dataset (Dataset): The whole Dataset.
            labels (Sequence[int]): The set of labels (targets) to keep.
        """
        self.dataset = dataset
        if isinstance(labels, Task):
            labels = labels.classes
        self.labels: Set[int] = set(labels)

        if isinstance(self.dataset, VisionDataset):
            from datasets.dataset import fix_vision_dataset
            fix_vision_dataset(self.dataset)

        # get the mask to select only the relevant items.
        mask = get_mask(self.dataset, self.labels)
        indices = mask.nonzero().flatten()

        if isinstance(dataset, SubsetBase):
            data = dataset.dataset.data[dataset.indices]
            targets = dataset.dataset.targets[dataset.indices]
        else:
            data = dataset.data
            targets = dataset.targets

        # only keep the elements where y is in `self.labels`
        self.data = torch.as_tensor(data)
        self.data = self.data[indices]
        
        self.targets = torch.as_tensor(targets)
        self.targets = self.targets[indices]

        shape_1 = self.dataset.data[0].shape
        shape_2 = self.data[0].shape
        assert shape_1 == shape_2, f"Shapes should be the same: {shape_1}, {shape_2}"
        
        # Add a "channel" dimension, if none exists.
        from utils.utils import fix_channels
        self.data = fix_channels(self.data)

        # Convert the samples to float, if not done already. 
        if self.data.dtype == torch.uint8:
            self.data = self.data.float() / 255

        super().__init__(self.data, self.targets)
        self.tensors: List[Tensor]

    def __add__(self, other: "ClassSubset") -> Union["ClassSubset", ConcatDataset]:  # type: ignore
        if isinstance(other, ClassSubset):
            assert self.dataset is other.dataset, "can't add subsets of different datasets"
            labels = list(set(self.labels).union(set(other.labels)))
            return ClassSubset(self.dataset, labels=labels)
        return super().__add__(other)

    def __str__(self) -> str:
        return f"VisionDatasetSubset of dataset of type {type(self.dataset).__name__} with labels {self.labels}."

    @contextmanager
    def without_labels(self):
        self.tensors = list(self.tensors)
        labels = self.tensors.pop()
        yield
        self.tensors.append(labels)
        self.tensors = tuple(self.tensors)


def get_mask(dataset: Dataset, labels: Iterable[int]) -> Tensor:
    """Returns a binary mask to select only the entries with a label within `labels` from `dataset`.

    To get the corresponding indices, apply `.nonzero()` to the result.

    Parameters
    ----------
    - dataset : Dataset
    
        A dataset, with a `targets` attribute containing the labels.
    - labels : Sequence[int]
    
        The labels (classes) to keep.
    
    Returns
    -------
    Tensor
        A boolean mask to select the values from the dataset.
    """
    selected_mask = torch.zeros(len(dataset), dtype=torch.bool)
    if isinstance(dataset, Subset):
        targets = dataset.targets
    elif isinstance(dataset, SubsetBase):
        targets = dataset.dataset.targets[dataset.indices]
    else:
        targets = dataset.targets
    if not torch.is_tensor(targets):
        targets = torch.as_tensor(targets)
    for label in labels:
        selected_mask |= (targets == label)
    return selected_mask
