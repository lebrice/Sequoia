from utils.logging_utils import get_logger
from typing import Iterable, Iterator, Sized, Tuple, Union, Set
from itertools import cycle
import numpy as np
import random
from torch import Tensor
from datasets.subset import ClassSubset, Subset
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import VisionDataset, CIFAR100
from torch.utils.data.sampler import SubsetRandomSampler, Sampler
import torch
from .subset import Subset
from torch._six import int_classes as _int_classes
logger = get_logger(__file__)


def get_lab_unlab_idxs(labels: Tensor, p: float = 0.2):
    #p - percentage of labeled data to be kept per class
    indices = np.arange(len(labels))
    classes = np.unique(labels)
    # Ensure uniform distribution of labels
    np.random.shuffle(indices)
    indices_train = [list(filter(lambda idx: labels[idx] == i, indices))[:] for i in classes]
    indices_train_lab = np.hstack([indices_train[i][:int(p*len(indices_train[i]))] for i in range(len(indices_train))])
    indices_train_unlab = np.hstack([indices_train[i][int(p*len(indices_train[i])):] for i in range(len(indices_train))])
    #lab and unlab indicies dont overlap
    indices_train_lab = torch.from_numpy(indices_train_lab)
    indices_train_unlab = torch.from_numpy(indices_train_unlab)

    return indices_train_lab, indices_train_unlab



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
    indices_train = torch.from_numpy(indices_train)
    indices_unlabelled = torch.from_numpy(indices_unlabelled)
    sampler_train = SubsetRandomSampler(indices_train)
    sampler_unlabelled = SubsetRandomSampler(indices_unlabelled)
    return sampler_train, sampler_unlabelled

def train_valid_split(train_dataset: VisionDataset, valid_dataset: VisionDataset, valid_fraction: float=0.2) -> Tuple[VisionDataset, VisionDataset]:
    n = len(train_dataset)
    valid_len: int = int((n * valid_fraction))
    train_len: int = n - valid_len
    
    indices = np.arange(n, dtype=int)
    np.random.shuffle(indices)
    
    valid_indices = indices[:valid_len]
    train_indices = indices[valid_len:]
    train = Subset(train_dataset, train_indices)
    valid = Subset(valid_dataset, valid_indices)
    logger.info(f"Training samples: {len(train)}, Valid samples: {len(valid)}")
    return train, valid


def unbatch(dataloader: Iterable[Tuple[Tensor, Tensor]]) -> Iterable[Tuple[Tensor, Tensor]]:
    """ Unbatches a dataloader.
    NOTE: this is a generator for a single pass through the dataloader, not multiple.
    """
    for batch in dataloader:
        x, y = batch
        yield from zip(x, y)

class zip_dataloaders(Iterable[Tuple[Tensor]], Sized):
    """ Given a DataLoader, returns an Iterable that drops the labels. """
    def __init__(self, dataloader_1: DataLoader, dataloader_2: DataLoader):
        from itertools import cycle
        if len(dataloader_1) > len(dataloader_2):
            self.dataloader_1 = dataloader_1
            self.dataloader_2 = cycle(dataloader_2)
            self._len = len(dataloader_1)
        elif len(dataloader_1) < len(dataloader_2):
            self.dataloader_1 = cycle(dataloader_1)
            self.dataloader_2 = dataloader_2
            self._len = len(dataloader_2)
        else:
            self.dataloader_1 = dataloader_1
            self.dataloader_2 = dataloader_2
            self._len = len(dataloader_2)

    def __iter__(self) -> Iterator[Tuple[Tensor]]: 
        for b1, b2 in zip(self.dataloader_1, self.dataloader_2):
            yield (b1, b2)

    def __len__(self) -> int:
        return self._len


class unlabeled(Iterable[Tuple[Tensor]], Sized):
    """ Given a DataLoader, returns an Iterable that drops the labels. """
    def __init__(self, labeled_dataloader: DataLoader):
        self.loader = labeled_dataloader

    def __iter__(self) -> Iterator[Tuple[Tensor]]:
        for x, y in self.loader:
            yield x,

    def __len__(self) -> int:
        return len(self.loader)



def keep_in_memory(dataset: VisionDataset) -> None:
    """ Converts the dataset's `data` and `targets` attributes to Tensors.
    
    This has the consequence of keeping the entire dataset in memory.
    """
    if not isinstance(dataset.data, (np.ndarray, Tensor)):
        dataset.data = torch.as_tensor(dataset.data)
    if not isinstance(dataset.targets, (np.ndarray, Tensor)):
        dataset.targets = torch.as_tensor(dataset.targets)

    if isinstance(dataset, CIFAR100):
        # TODO: Cifar100 seems to want its 'data' to a numpy ndarray. 
        dataset.data = np.asarray(dataset.data)


class RandomSampler_Semi(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices_l (sequence): a sequence of indices of labeled samples
        indices_ul (sequence): a sequence of indices of unlabeled samples
    """

    def __init__(self, indices_l, indices_ul):
        #make lists same size
        self.indices_l, self.indices_ul = self.twolists(indices_l, indices_ul)
    
    def twolists(self, l1, l2):
        a1 = len(l1)
        a2 = len(l2)
        if a1>a2:
            l = np.array(list(zip(l1,cycle(l2))))
            l2 = l[:,1].tolist()
        else:
            l = np.array(list(zip(cycle(l1),l2)))
            l1 = l[:,0].tolist()        
        return l1, l2
    
    def __iter__(self):
        random.shuffle(self.indices_l)
        random.shuffle(self.indices_ul)
        return iter(list(zip(self.indices_l, self.indices_ul))) #(self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return max(len(self.indices_l), len(self.indices_ul))


class BatchSampler_SemiUpservised(Sampler):
    r"""Batch sampler for semi-supervised loaders
    """
    def __init__(self,  indx_labeled:list, indx_unlabeled:list, batch_size, drop_last=False):

        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))

        self.sampler = RandomSampler_Semi(indx_labeled, indx_labeled)
        self._len = len(self.sampler)

        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx_l, idx_ul in self.sampler:
            batch.append(idx_l)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
            batch.append(idx_ul)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return self._len // (self.batch_size//2)
        else:
            return (self._len + (self.batch_size//2) - 1) // (self.batch_size//2)

class SemiSupervisedDataset(Dataset):
    def __init__(self, dataset:Dataset, indices_labeled: Set, indices_unlabeled: Set):
        self.dataset = dataset #labeled dataset
        self.indices_labeled = indices_labeled
        self.indices_unlabeled = indices_unlabeled
    
    def __getitem__(self, idx):
        if idx in self.indices_labeled:
            return self.dataset[idx]
        elif idx in self.indices_unlabeled:
            x, y = self.dataset[idx]
            return x, 
        else:
            raise IndexError

    def __len__(self):
        return len(self.dataset)

class FlexiTransformDataset(Dataset):
    '''
    Dataset for which transformations can be changed dynamically.
    '''
    def __init__(self, dataset:Dataset, transfrom):
        self.dataset = dataset #labeled dataset
        self.transform = transfrom
    
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return self.transform(x), y
    
    def set_transformation_mode(self, transform):
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

class TransformedDataset(Dataset):
    '''
    Adds transfromations to existing labeled dataset
    '''
    def __init__(self, dataset:Dataset, transform=None, target_transform=None):
        self.dataset = dataset #labeled dataset
        self.transform = transform
        self.target_transform = target_transform
    
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)
        return x, y
    
    def __len__(self):
        return len(self.dataset)