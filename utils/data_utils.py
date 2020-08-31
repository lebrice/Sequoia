import os
from pathlib import Path
from typing import Dict, Iterable, Iterator, Sized, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR100, VisionDataset

from utils.logging_utils import get_logger

logger = get_logger(__file__)


def train_valid_split(train_dataset: VisionDataset, valid_fraction: float=0.2) -> Tuple[VisionDataset, VisionDataset]:
    n = len(train_dataset)
    valid_len: int = int((n * valid_fraction))
    train_len: int = n - valid_len
    
    indices = np.arange(n, dtype=int)
    np.random.shuffle(indices)
    
    valid_indices = indices[:valid_len]
    train_indices = indices[valid_len:]
    train = Subset(train_dataset, train_indices)
    valid = Subset(train_dataset, valid_indices)
    logger.info(f"Training samples: {len(train)}, Valid samples: {len(valid)}")
    return train, valid


def unbatch(dataloader: Iterable[Tuple[Tensor, Tensor]]) -> Iterable[Tuple[Tensor, Tensor]]:
    """ Unbatches a dataloader.
    NOTE: this is a generator for a single pass through the dataloader, not multiple.
    """
    for batch in dataloader:
        tensors = batch
        yield from (zip(*tensors) if isinstance(tensors, tuple) else tensors)


class unlabeled(Iterable[Tuple[Tensor]], Sized):
    """ Given a DataLoader, returns an Iterable that drops the labels. """
    def __init__(self, labeled_dataloader: DataLoader):
        self.loader = labeled_dataloader

    def __iter__(self) -> Iterator[Tuple[Tensor]]:
        for batch in self.loader:
            assert isinstance(batch, tuple)
            x = batch[0]
            yield x,

    def __len__(self) -> int:
        return len(self.loader)


def keep_in_memory(dataset: VisionDataset) -> None:
    """ Converts the dataset's `data` and `targets` attributes to Tensors.
    
    This has the consequence of keeping the entire dataset in memory.
    """

    if hasattr(dataset, "data") and not isinstance(dataset.data, (np.ndarray, Tensor)):
        dataset.data = torch.as_tensor(dataset.data)
    if not isinstance(dataset.targets, (np.ndarray, Tensor)):
        dataset.targets = torch.as_tensor(dataset.targets)

    if isinstance(dataset, CIFAR100):
        # TODO: Cifar100 seems to want its 'data' to a numpy ndarray. 
        dataset.data = np.asarray(dataset.data)


class FixChannels(nn.Module):
    """ Transform that fixes the number of channels in input images. 
    
    For instance, if the input shape is:
    [28, 28] -> [3, 28, 28] (copy the image three times)
    [1, 28, 28] -> [3, 28, 28] (same idea)
    [10, 1, 28, 28] -> [10, 3, 28, 28] (keep batch intact, do the same again.)
    
    """
    def __call__(self, x: Tensor) -> Tensor:
        if x.ndim == 2:
            x = x.reshape([1, *x.shape])
            x = x.repeat(3, 1, 1)
        if x.ndim == 3 and x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        if x.ndim == 4 and x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return x


def get_imagenet_location() -> Path:
    from socket import gethostname
    hostname = gethostname()
    # For each hostname prefix, the location where the torchvision ImageNet dataset can be found.
    # TODO: Add the location for your own machine.
    imagenet_locations: Dict[str, Path] = {
        "mila": Path("/network/datasets/imagenet.var/imagenet_torchvision"),
        "": Path("/network/datasets/imagenet.var/imagenet_torchvision"),
    }
    for prefix, v in imagenet_locations.items():
        if hostname.startswith(prefix):
            return v
    if "IMAGENET_DIR" in os.environ:
        return Path(os.environ["IMAGENET_DIR"])
    raise RuntimeError(
        f"Could not find the ImageNet dataset on this machine with hostname "
        f"{hostname}. Known <prefix --> location> pairs: {imagenet_locations}"
    )
