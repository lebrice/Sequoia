import torch
import numpy as np
from itertools import repeat, cycle
from torch.utils.data.sampler import SubsetRandomSampler

def get_sampler(labels, p:float=None):
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
