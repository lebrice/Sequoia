from contextlib import contextmanager
from typing import Iterable, List, Sequence, Set, Tuple, Union

import torch
from PIL import Image as image
from PIL.Image import Image
from torch import Tensor
from torch.utils.data import TensorDataset
from torchvision.datasets import VisionDataset
from torchvision.transforms import Normalize


class VisionDatasetSubset(TensorDataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        classes (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset: VisionDataset, labels: Sequence[int]):
        self.dataset = dataset
        self.labels: Set[int] = set(labels)
        indices = get_mask(self.dataset, self.labels).nonzero()
        
        self.data = self.dataset.data[indices]
        self.targets = self.dataset.targets[indices]

        if self.data.dtype == torch.uint8:
            self.data = self.data.float() / 255

        super().__init__(self.data, self.targets)
        self.tensors: List[Tensor]

    def __add__(self, other: "VisionDatasetSubset") -> "VisionDatasetSubset":  # type: ignore
        assert self.dataset is other.dataset, "can't add subsets of different datasets"
        labels = list(set(self.labels).union(set(other.labels)))
        return VisionDatasetSubset(self.dataset, labels=labels)
    
    def __str__(self) -> str:
        return f"VisionDatasetSubset of dataset of type {type(self.dataset).__name__} with labels {self.labels}."

    @contextmanager
    def without_labels(self):
        self.tensors = list(self.tensors)
        labels = self.tensors.pop()
        yield
        self.tensors.append(labels)
        self.tensors = tuple(self.tensors)


def get_mask(dataset: VisionDataset, labels: Iterable[int]) -> Tensor:
    """Returns a binary mask to select only the entries with a label within `labels` from `dataset`.

    To get the corresponding indices, apply `.nonzero()` to the result.

    Parameters
    ----------
    - dataset : VisionDataset
    
        A dataset, with a `targets` attribute containing the labels.
    - labels : Sequence[int]
    
        The labels (classes) to keep.
    
    Returns
    -------
    Tensor
        A boolean mask to select the values from the dataset.
    """
    selected_mask = torch.zeros(len(dataset), dtype=torch.bool)
    for label in labels:
        selected_mask |= (dataset.targets == label)
    return selected_mask
