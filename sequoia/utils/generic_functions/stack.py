""" Generic function for concatenating ndarrays/tensors/distributions/Mappings
etc.
"""
from collections.abc import Mapping
from functools import singledispatch
from typing import Any, Dict, List, Sequence, Union, TypeVar, overload

import numpy as np
import torch
from gym import Space, spaces
from sequoia.common.spaces.sparse import Sparse
from sequoia.utils.categorical import Categorical
from torch import Tensor
from sequoia.common.spaces.utils import get_batch_type_for_item_type, get_item_type_for_batch_type

T = TypeVar("T")


# @overload
# def stack(first_item: List[T]) -> Sequence[T]:
#     ...

# @overload
# def stack(first_item: T, *others: T) -> Sequence[T]:
#     ...


@singledispatch
def stack(first_item: Union[T, Sequence[T]], *others: T, **kwargs) -> Any:
    # By default, if we don't know how to handle the item type, just
    # return an ndarray with with all the items.
    # note: We could also try to return a tensor, rather than an ndarray
    # but I'd rather keep it simple for now.
    if not others:
        # If this was called like stack(tensor_list), then we just split off
        # the list of items.
        if first_item is None:
            # Stacking a list of 'None' items returns None.
            return None
        # assert isinstance(first_item, (list, tuple)), first_item
        # assert len(first_item) > 1, first_item
        items = first_item
        return stack(items[0], *items[1:], **kwargs)
    np_stack_kwargs = kwargs.copy()
    if "dim" in np_stack_kwargs:
        np_stack_kwargs["axis"] = np_stack_kwargs.pop("dim")
    return np.stack([first_item, *others], **np_stack_kwargs)


@stack.register(type(None))
def _stack_none(first_item: None, *others: None, **kwargs) -> Union[None, np.ndarray]:
    # TODO: Should we return an ndarray with 'None' entries, of dtype np.object_? or
    # just a single None?
    # Opting for a single None for now, as it's easier to work with. (`v is None` works)
    if all(v is None for v in others):
        return None
    return np.array([first_item, *others])
    # if not others:
    #     return None
    # return np.array([None, *others])


@stack.register(np.ndarray)
def _stack_ndarrays(first_item: np.ndarray, *others: np.ndarray, **kwargs) -> np.ndarray:
    return np.stack([first_item, *others], **kwargs)


@stack.register(Tensor)
def _stack_tensors(first_item: Tensor, *others: Tensor, **kwargs) -> Tensor:
    return torch.stack([first_item, *others], **kwargs)


@stack.register(Mapping)
def _stack_dicts(first_item: Dict, *others: Dict, **kwargs) -> Dict:
    # Check if the stack should use a different dtype than the items. If not, then use the same type
    # as the items. 
    stack_dtype = get_batch_type_for_item_type(type(first_item)) or type(first_item)
    return stack_dtype(
        **{
            key: stack(first_item[key], *(other[key] for other in others), **kwargs)
            for key in first_item.keys()
        }
    )


@stack.register(Categorical)
def _stack_distributions(first_item: Categorical, *others: Categorical, **kwargs) -> Categorical:
    # NOTE: Could use `expand` rather than `stack` if all the logits/log_probs are the same.
    return Categorical(
        logits=torch.stack([first_item.logits, *(other.logits for other in others)], **kwargs)
    )


@singledispatch
def unstack(stacked_values: Union[Sequence[T], Any]) -> Sequence[T]:
    """ Does the inverse operation of 'stack'. """
    raise NotImplementedError(f"Don't know how to unstack values of type {stacked_values}")


from collections import abc as _abc


@unstack.register(_abc.Sequence)
@unstack.register(np.ndarray)
@unstack.register(Tensor)
def _unstack_arraylike(stacked_v: Sequence[T]) -> List[T]:
    return list(stacked_v)


K = TypeVar("K")


@unstack.register(_abc.Mapping)
def _unstack_dicts(stacked_v: Dict[K, Sequence[T]]) -> List[Dict[K, T]]:
    if not stacked_v:
        raise RuntimeError(f"Can't unstack an empty dict.")
    keys = list(stacked_v.keys())
    unstacked_dicts = {k: unstack(stacked_values) for k, stacked_values in stacked_v.items()}

    lengths_of_unstacked_values = {k: len(values) for k, values in unstacked_dicts.items()}
    if len(set(lengths_of_unstacked_values.values())) != 1:
        raise RuntimeError(
            f"All unstacked values should have the same length, but got different lengths instead: "
            f"{lengths_of_unstacked_values}"
        )
    _, n_values = lengths_of_unstacked_values.popitem()
    # Check if we shoudl use a type in particular for items, otherwise use the same as the batch.
    item_type = get_item_type_for_batch_type(type(stacked_v)) or type(stacked_v)
    return [
        item_type(
            **{k: unstacked_values[i] for k, unstacked_values in unstacked_dicts.items()}
        )
        for i in range(n_values)
    ]


@unstack.register(Categorical)
def _unstack_distribution(stacked_dist: Categorical) -> List[Categorical]:
    return [Categorical(logits=stacked_dist.logits[i]) for i in range(stacked_dist.logits.shape[0])]
