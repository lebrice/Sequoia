from utils.logging_utils import get_logger
from typing import Iterable, Iterator, Sized, Tuple

import numpy as np
from torch import Tensor, from_numpy
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset

from .subset import Subset

logger = get_logger(__file__)

def get_semi_sampler(labels, p:float=None):
    #p - percentage of labeled data to be kept
    #print(type(labels))
    indices = np.arange(len(labels))
    classes = np.unique(labels)
    # Ensure uniform distribution of labels
    np.random.shuffle(indices)

    indices_train = [list(filter(lambda idx: labels[idx] == i, indices))[:] for i in classes]
    indices_train = np.hstack([indices_train[i][:int(p*len(indices_train[i]))] for i in range(len(indices_train))])
    indices_unlabelled = np.hstack(
        [list(filter(lambda idx: labels[idx] == i, indices))[:] for i in classes])
    # print (indices_train.shape)
    # print (indices_valid.shape)
    # print (indices_unlabelled.shape)
    indices_train = from_numpy(indices_train)   
    indices_unlabelled = from_numpy(indices_unlabelled)
    sampler_train = SubsetRandomSampler(indices_train)
    sampler_unlabelled = SubsetRandomSampler(indices_unlabelled)
    return sampler_train, sampler_unlabelled
    
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
