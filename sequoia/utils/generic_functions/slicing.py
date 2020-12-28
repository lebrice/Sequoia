""" Extendable utility functions for getting and settings slices of arbitrarily
nested objects.

"""
from functools import singledispatch
from typing import Any, Dict, List, Sequence, Tuple, TypeVar

import numpy as np
import torch
from torch import Tensor

from ._namedtuple import is_namedtuple, NamedTuple

K = TypeVar("K")
V = TypeVar("V")
T = TypeVar("T")


@singledispatch
def get_slice(value: Any, indices: Sequence[int]) -> Any:
    """ Returns a slices of `value` at the given indices.
    """
    if value is None:
        return None
    return value[indices]


@get_slice.register(dict)
def _get_dict_slice(value: Dict[K, V], indices: Sequence[int]) -> Dict[K, V]:
    return type(value)(
        (k, get_slice(v, indices)) for k, v in  value.items()
    ) 


@get_slice.register(NamedTuple)
def _get_namedtuple_slice(value: Tuple[T, ...], indices: Sequence[int]) -> Tuple[T, ...]:
    # NOTE: we use type(value)( ... ) to create the output dicts or tuples, in
    # case a subclass of tuple or dict is being used (e.g. NamedTuples). 
    return type(value)(*[
        get_slice(v, indices) for v in value
    ])


@get_slice.register(tuple)
def _get_tuple_slice(value: Tuple[T, ...], indices: Sequence[int]) -> Tuple[T, ...]:
    # NOTE: we use type(value)( ... ) to create the output dicts or tuples, in
    # case a subclass of tuple or dict is being used (e.g. NamedTuples). 
    return type(value)([
        get_slice(v, indices) for v in value
    ])


@get_slice.register(NamedTuple)
def _get_namedtuple_slice(value: Tuple[T, ...], indices: Sequence[int]) -> Tuple[T, ...]:
    # NOTE: we use type(value)( ... ) to create the output dicts or tuples, in
    # case a subclass of tuple or dict is being used (e.g. NamedTuples). 
    return type(value)(*[
        get_slice(v, indices) for v in value
    ])



@singledispatch
def set_slice(target: Any, indices: Sequence[int], values: Sequence[Any]) -> None:
    """ Sets `values` at positions `indices` in `target`.
    
    Modifies the `target` in-place.
    """
    target[indices] = values


@set_slice.register(dict)
def _set_dict_slice(target: Dict[K, Sequence[V]], indices: Sequence[int], values: Dict[K, Sequence[V]]) -> None:
    for key, target_values in target.items():
        set_slice(target_values, indices, values[key])


@set_slice.register(tuple)
def _set_tuple_slice(target: Tuple[T, ...], indices: Sequence[int], values: Tuple[T, ...]) -> None:
    assert isinstance(values, tuple)
    assert len(target) == len(values)
    for target_item, values_item in zip(target, values):
        set_slice(target_item, indices, values_item)


# TODO: Remove this, unless it gets used somewhere.
# from collections import namedtuple
# from typing import NamedTuple


# @singledispatch
# def concatenate(value_a: T, value_b: T, *args, **kwargs) -> T:
#     """ Concatenates two tensors. Also supports arbitrarily nested tuples and
#     dicts of tensors.
#     """
#     raise NotImplementedError(
#         f"Function `{concatenate}` doesn't have a registered implementation "
#         f"for handling values of type {type(value_a)}. "
#         f"(value_a: {value_a}, value_b: {value_b}, args: {args}, kwargs: {kwargs})."
#     )

# @concatenate.register(np.ndarray)
# def _concatenate_numpy_arrays(value_a: np.ndarray, value_b: np.ndarray, *args, **kwargs) -> np.ndarray:
#     return np.concatenate([value_a, value_b], *args, **kwargs)


# @concatenate.register(Tensor)
# def _concatenate_tensors(value_a: Tensor, value_b: Tensor, *args, **kwargs) -> np.ndarray:
#     return torch.concatenate([value_a, value_b], *args, **kwargs)


# @concatenate.register(list)
# def _concatenate_lists(value_a: List[T], value_b: List[T], *args, **kwargs) -> np.ndarray:
#     # TODO: Should we concatenate lists like usual? or should we treat those
#     # like we do tuples below and concatenate the elements?
#     raise NotImplementedError(
#         f"Refusing to concatenate lists, because its ambiguous atm wether this "
#         f"should concat the elements of the two lists or the lists themselves. "
#     )
#     return np.concatenate([value_a, value_b], *args, **kwargs)


# @concatenate.register(NamedTuple)
# def _concatenate_namedtuples(value_a: Tuple[T, ...], value_b: Tuple[T, ...], *args, **kwargs) -> Tuple[T, ...]:
#     assert type(value_a) == type(value_b)
#     assert len(value_a) == len(value_b)
#     return type(value_a)(*[
#         concatenate(value_a[i], value_b[i], *args, **kwargs)
#         for i in range(len(value_a))
#     ])


# @concatenate.register(tuple)
# def _concatenate_tuples(value_a: Tuple[T, ...], value_b: Tuple[T, ...], *args, **kwargs) -> Tuple[T, ...]:
#     assert type(value_a) == type(value_b)
#     assert len(value_a) == len(value_b)

#     if is_namedtuple(value_a):
#         return _concatenate_namedtuples(value_a, value_b, *args, **kwargs)

#     return type(value_a)(
#         concatenate(value_a[i], value_b[i], *args, **kwargs)
#         for i in range(len(value_a))
#     )


# @concatenate.register(dict)
# def _concatenate_dicts(value_a: Dict[K, V], value_b: Dict[K, V], *args, **kwargs) -> Dict[K, V]:
#     assert type(value_a) == type(value_b)
#     assert value_a.keys() == value_b.keys()
#     return type(value_a)(
#         (key, concatenate(value_a[key], value_b[key], *args, **kwargs))
#         for key in value_a
#     )
