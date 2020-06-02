from contextlib import contextmanager
from typing import *

import numpy as np
import torch
from PIL import Image as image
from PIL.Image import Image
from torch import Tensor
from torch.utils.data import TensorDataset, Dataset, Subset as SubsetBase
from torchvision.transforms import Normalize
from torchvision.datasets.vision import VisionDataset
from common.task import Task
import logging
logger = logging.getLogger(__file__)

D = TypeVar("D", bound=Dataset)
V = TypeVar("V", bound=VisionDataset)


class Subset(Generic[D], Dataset):
    r"""Subset of a dataset at specified indices or using the specified classes.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (Union[Tensor, Sequence[int]]): Indices in the whole set selected for subset
        indices (Union[Task, Sequence[int]]): Task object or list of classes to keep.
    """
    def __init__(self, dataset: D,
                       indices: Union[Tensor, Sequence[int]]=None,
                       classes: Union[Task, Sequence[int]]=None):
        self.dataset: D = dataset
        assert (indices is not None) != (classes is not None), f"Use either indices or classes"

        if isinstance(indices, Task) and classes is None:
            logger.warning(UserWarning("Set `classes` kwarg instead of passing just a `Task` as positional!"))
            classes = indices

        if classes:
            if isinstance(classes, Task):
                classes = classes.classes
            mask = get_mask(dataset, classes)
            indices = mask.nonzero().flatten()

        self.indices = torch.as_tensor(indices)

    def __getitem__(self, idx: Union[int, List[int], np.ndarray, Tensor]) -> Tuple[Tensor, Optional[Tensor]]:
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class VisionDatasetSubset(Subset[VisionDataset]):
    """ Subset of a VisionDataset, with the `data` and `target` attributes.

    Can contain either a subset of the indices or of the classes of the dataset.

    When only passed a regular `torch.utils.data.Subset` object which has a
    VisionDataset as a `dataset` attribute, and no `indices` or `classes`, will
    "convert" it into a new `VisionDatasetSubset` object.
    """
    def __init__(self, dataset: VisionDataset,
                       indices: Union[Tensor, Sequence[int]]=None,
                       classes: Union[Sequence[int], Task]=None):
        if isinstance(dataset, SubsetBase) and indices is None and classes is None and isinstance(dataset.dataset, VisionDataset):
            # Convert a `SubsetBase` of a VisionDataset into a VisionDatasetSubset
            indices = dataset.indices
            dataset = dataset.dataset
        
        if not isinstance(dataset, (VisionDataset, VisionDatasetSubset)):
            raise RuntimeError(f"VisionDatasetSubset should be called with a VisionDataset as argument, not {dataset}")
        super().__init__(dataset, indices=indices, classes=classes)

        # TODO: This doesn't really belong in this class. Could maybe move it somewhere else?
        # When `True`, will drop the labels and return (x, None) instead of (x, y)
        self._drop_labels: bool = False

    @property
    def data(self) -> Tensor:
        return self.dataset.data[self.indices]
    
    @property
    def targets(self) -> Tensor:
        return self.dataset.targets[self.indices]

    def __getitem__(self, idx: Union[int, List[int], np.ndarray, Tensor]) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        x, y = self.dataset[self.indices[idx]]
        if self._drop_labels:
            return x
        return x, y

    @contextmanager
    def without_labels(self):
        """
        Context manager that temporarily stops yielding labels.
        """
        val = self._drop_labels
        self._drop_labels = True
        yield
        self._drop_labels = val


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
    logger.debug(f"Getting mask for dataset: {dataset}, labels: {labels}")
    
    if isinstance(dataset, VisionDataset):
        targets = dataset.targets
    elif isinstance(dataset, VisionDatasetSubset):
        targets = dataset.targets
    elif isinstance(dataset, TensorDataset):
        targets = dataset.tensors[-1]
    elif isinstance(dataset, Dataset):
        # TODO: See what would happen if this is actually used (with something like ImageNet maybe?)
        logger.warning(f"Contructing the labels tensor by looping over the dataset! ")
        loader = torch.utils.data.DataLoader(dataset, batch_size=10)
        ys: List[int] = []
        for i, sample in enumerate(loader):
            if isinstance(sample, tuple) and len(sample) > 1:
                y = sample[-1]
            else:
                raise RuntimeError(f"Don't know how to deal with dataset sample at index {i}: {sample}")
            y = np.asarray(y, dtype=int).flatten()
            ys.extend(y)
        targets = np.asarray(ys)
    else:
        raise RuntimeError(f"Don't know how to access the labels of dataset {dataset}")
    
    if not torch.is_tensor(targets):
        targets = torch.as_tensor(targets)

    selected_mask = torch.zeros(len(targets), dtype=torch.bool)
    for label in labels:
        selected_mask |= (targets == label)
    return selected_mask
