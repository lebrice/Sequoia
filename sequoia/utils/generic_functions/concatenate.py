""" Generic function for concatenating ndarrays/tensors/distributions/Mappings
etc.

Extremely similar to `stack.py`, but concatenates along the described axis.
"""

from collections.abc import Mapping
from functools import singledispatch
from typing import Any, Dict, List, Sequence, Union, TypeVar, overload

import numpy as np
import torch
from gym import Space, spaces
from sequoia.common.spaces import Sparse
from sequoia.utils.categorical import Categorical
from torch import Tensor

T = TypeVar("T")


# @overload
# def concatenate(first_item: List[T], **kwargs) -> Sequence[T]:
#     ...

# @overload
# def concatenate(first_item: T, *others: T, **kwargs) -> Sequence[T]:
#     ...


@singledispatch
def concatenate(first_item: Union[T, List[T]], *others: T, **kwargs) -> Union[Sequence[T], Any]:
    # By default, if we don't know how to handle the item type, just
    # returns an ndarray with with all the items.

    if not others:
        # If this was called like concatenate(tensor_list), then we just split off
        # the list of items.
        assert isinstance(first_item, (list, tuple))
        assert len(first_item) > 1
        items = first_item
        return concatenate(items[0], *items[1:], **kwargs)

    return np.asarray([first_item, *others], **kwargs)


@concatenate.register(np.ndarray)
def _concatenate_ndarrays(first_item: np.ndarray, *others: np.ndarray, **kwargs) -> np.ndarray:
    if not first_item.shape:
        # can't concatenate 0-dimensional arrays, so we stack them instead:
        return np.stack([first_item, *others], **kwargs)
    return np.concatenate([first_item, *others], **kwargs)


@concatenate.register(Tensor)
def _concatenate_tensors(first_item: Tensor, *others: Tensor, **kwargs) -> Tensor:
    if not first_item.shape:
        # can't concatenate 0-dimensional tensors, so we stack them instead.
        return torch.stack([first_item, *others], **kwargs)
    return torch.cat([first_item, *others], **kwargs)


@concatenate.register(Mapping)
def _concatenate_dicts(first_item: Dict, *others: Dict, **kwargs) -> Dict:
    return type(first_item)(**{
        key: concatenate(first_item[key], *(other[key] for other in others), **kwargs)
        for key in first_item.keys()
    })


@concatenate.register(Categorical)
def _concatenate_distributions(first_item: Categorical, *others: Categorical, **kwargs) -> Categorical:
    return Categorical(logits=torch.cat([
        first_item.logits, *(other.logits for other in others)
    ], *kwargs))


from continuum import TaskSet
from continuum.tasks import concat

@concatenate.register
def _concatenate_tasksets(first_item: TaskSet, *others: TaskSet) -> TaskSet:
    return concat([first_item, *others])


from torch.utils.data import Dataset, ConcatDataset


@concatenate.register
def _concatenate_datasets(first_item: Dataset, *others: Dataset) -> TaskSet:
    return ConcatDataset([first_item, *others])