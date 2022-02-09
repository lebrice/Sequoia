""" Extendable utility functions for getting and settings slices of arbitrarily
nested objects.

"""
from functools import singledispatch
from typing import Any, Dict, Sequence, Tuple, TypeVar

import numpy as np
from torch import Tensor

from ._namedtuple import is_namedtuple

K = TypeVar("K")
V = TypeVar("V")
T = TypeVar("T")


@singledispatch
def get_slice(value: T, indices: Sequence[int]) -> T:
    """Returns a slices of `value` at the given indices."""
    if value is None:
        return None
    return value[indices]


@get_slice.register(dict)
def _get_dict_slice(value: Dict[K, V], indices: Sequence[int]) -> Dict[K, V]:
    return type(value)((k, get_slice(v, indices)) for k, v in value.items())


@get_slice.register(tuple)
def _get_tuple_slice(value: Tuple[T, ...], indices: Sequence[int]) -> Tuple[T, ...]:
    # NOTE: we use type(value)( ... ) to create the output dicts or tuples, in
    # case a subclass of tuple or dict is being used (e.g. NamedTuples).
    if is_namedtuple(value):
        return type(value)(*[get_slice(v, indices) for v in value])
    return type(value)([get_slice(v, indices) for v in value])


@singledispatch
def set_slice(target: Any, indices: Sequence[int], values: Sequence[Any]) -> None:
    """Sets `values` at positions `indices` in `target`.

    Modifies the `target` in-place.
    """
    target[indices] = values


from sequoia.utils.categorical import Categorical


@set_slice.register
def _set_slice_categorical(
    target: Categorical, indices: Sequence[int], values: Sequence[Any]
) -> None:
    target.logits[indices] = values.logits


@set_slice.register(np.ndarray)
def _set_slice_ndarray(target: np.ndarray, indices: Sequence[int], values: Sequence[Any]) -> None:
    if isinstance(indices, Tensor):
        indices = indices.cpu().numpy()
    if isinstance(values, Tensor):
        values = values.cpu().numpy()
    target[indices] = values


@set_slice.register(Tensor)
def _set_slice_ndarray(target: Tensor, indices: Sequence[int], values: Sequence[Any]) -> None:
    target[indices] = values


@set_slice.register(dict)
def _set_dict_slice(
    target: Dict[K, Sequence[V]], indices: Sequence[int], values: Dict[K, Sequence[V]]
) -> None:
    for key, target_values in target.items():
        set_slice(target_values, indices, values[key])


@set_slice.register(tuple)
def _set_tuple_slice(target: Tuple[T, ...], indices: Sequence[int], values: Tuple[T, ...]) -> None:
    assert isinstance(values, tuple)
    assert len(target) == len(values)
    for target_item, values_item in zip(target, values):
        set_slice(target_item, indices, values_item)
