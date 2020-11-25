""" Extendable utility functions for getting and settings slices of arbitrarily
nested objects. """
from functools import singledispatch
from typing import Dict, Tuple, TypeVar, Sequence, Any

K = TypeVar("K")
V = TypeVar("V")
T = TypeVar("T")


@singledispatch
def get_slice(value: Any, indices: Sequence[int]) -> Any:
    """ Returns a slices of `value` at the given indices.
    """
    return value[indices]


@get_slice.register(dict)
def get_dict_slice(value: Dict[K, V], indices: Sequence[int]) -> Dict[K, V]:
    return type(value)(
        (k, get_slice(v, indices)) for k, v in  value.items()
    ) 


@get_slice.register(tuple)
def get_tuple_slice(value: Tuple[T, ...], indices: Sequence[int]) -> Tuple[T, ...]:
    return type(value)(
        get_slice(v, indices) for v in value
    )


@singledispatch
def set_slice(target: Any, indices: Sequence[int], values: Sequence[Any]) -> None:
    """ Sets `values` at positions `indices` in `target`.
    
    Modifies the `target` in-place.
    """
    target[indices] = values


@set_slice.register(dict)
def set_dict_slice(target: Dict[K, Sequence[V]], indices: Sequence[int], values: Dict[K, Sequence[V]]) -> None:
    for key, target_values in target.items():
        set_slice(target_values, indices, values[key])


@set_slice.register(tuple)
def set_tuple_slice(target: Tuple[T, ...], indices: Sequence[int], values: Tuple[T, ...]) -> None:
    assert isinstance(values, tuple)
    assert len(target) == len(values)
    for target_item, values_item in zip(target, values):
        set_slice(target_item, indices, values_item)
