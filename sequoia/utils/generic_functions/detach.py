from collections.abc import Mapping
from functools import singledispatch
from typing import Any, Dict, Sequence, TypeVar

import numpy as np

from ._namedtuple import NamedTuple

T = TypeVar("T")

@singledispatch
def detach(value: T) -> T:
    """ Detaches a value when possible, else returns the value unchanged. """
    if hasattr(value, "detach") and callable(value.detach):
        return value.detach()
    raise NotImplementedError(f"Don't know how to detach value {value}!")
    # else:
    #     return value

@detach.register(np.ndarray)
@detach.register(type(None))
@detach.register(str)
@detach.register(int)
@detach.register(bool)
@detach.register(float)
def no_op_detach(v: Any) -> Any:
    return v


@detach.register(list)
@detach.register(tuple)
@detach.register(set)
def _detach_sequence(x: Sequence[T]) -> Sequence[T]:
    return type(x)(detach(v) for v in x)


@detach.register(NamedTuple)
def _detach_namedtuple(x: NamedTuple) -> NamedTuple:
    return type(x)(*[detach(v) for v in x])


@detach.register(Mapping)
def _detach_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """ Detaches all the keys and tensors in a dict, as well as all nested dicts.
    """
    return type(d)(**{
        detach(k): detach(v)
        for k, v in d.items()
    })
