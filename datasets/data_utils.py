from utils.logging_utils import get_logger
from typing import Iterable, Iterator, Sized, Tuple

import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset

from .subset import Subset

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
        x, y = batch
        yield from zip(x, y)


class unlabeled(Iterable[Tuple[Tensor]], Sized):
    """ Given a DataLoader, returns an Iterable that drops the labels. """
    def __init__(self, labeled_dataloader: DataLoader):
        self.loader = labeled_dataloader

    def __iter__(self) -> Iterator[Tuple[Tensor]]:
        for x, y in self.loader:
            yield x,

    def __len__(self) -> int:
        return len(self.loader)
