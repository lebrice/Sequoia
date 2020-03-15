from itertools import starmap
from typing import Iterable, Sequence, Set, Tuple

import torch
from PIL import Image as image
from PIL.Image import Image
from torch import Tensor
from torch.utils.data import Subset
from torchvision.datasets import VisionDataset
from torchvision.transforms import Normalize


class VisionDatasetSubset:
    r"""
    Subset of a dataset at specified indices.

    # TODO: Somehow, there's something that is getting messed up with the transforms.
    # TODO: The Mnist dataset would re-convert the tensors to PIL images, which is ugly and potentially slow.

    Arguments:
        dataset (Dataset): The whole Dataset
        classes (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset: VisionDataset, labels: Sequence[int]):
        self.dataset = dataset
        self.labels: Set[int] = set(labels)
        self.indices = get_mask(self.dataset, self.labels).nonzero()
        self.shuffle()
        self.skip_transforms: bool = False
    
    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        if self.skip_transforms:
            if img.dtype == torch.uint8:
                img = img.float() / 255
            return img, target
        
        if isinstance(idx, slice):
            results = list(starmap(self.process, zip(img, target)))
            imgs = torch.stack([r[0] for r in results])
            targets = torch.cat([r[1] for r in results])
            return imgs, targets

        return self.process(img, target)

    @property
    def data(self) -> Tensor:
        return self.dataset.data[self.indices]

    @property
    def targets(self) -> Tensor:
        return self.dataset.targets[self.indices]

    def process(self, img: Tensor, target: Tensor) -> Tuple[Image, Tensor]:
        if img.dim() == 3 and img.shape[0] == 1:
            img = img.view(*img.shape[1:])

        ## Taken from MNIST class in torchvision:
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = image.fromarray(img.numpy(), mode='L')
    
        if self.dataset.transform is not None:
            img = self.dataset.transform(img)
        if self.dataset.target_transform is not None:
            target = self.dataset.target_transform(target)
        ##
        return img, target

    def __len__(self) -> int:
        return len(self.indices)

    def shuffle(self) -> None:
        perm = torch.randperm(len(self))
        self.indices = self.indices[perm]
    
    def __add__(self, other: "VisionDatasetSubset") -> "VisionDatasetSubset":
        assert self.dataset is other.dataset, "can't add subsets of different datasets"
        labels = list(set(self.labels).union(set(other.labels)))
        return VisionDatasetSubset(self.dataset, labels=labels)
    

    def __str__(self) -> str:
        return f"VisionDatasetSubset of dataset of type {type(self.dataset).__name__} with labels {self.labels}."


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
    selected_mask = torch.zeros_like(dataset.targets, dtype=torch.bool)
    for label in labels:
        selected_mask |= (dataset.targets == label)
    return selected_mask
