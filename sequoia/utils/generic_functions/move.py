"""Defines a singledispatch function to move objects to a given device.
"""
from functools import singledispatch
from typing import Any, Dict, Sequence, TypeVar, Union
from sequoia.utils.generic_functions._namedtuple import is_namedtuple
import torch

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


@singledispatch
def move(x: T, device: Union[str, torch.device]) -> T:
    """Moves x to the specified device if possible, else returns x unchanged.
    NOTE: This works for Tensors or any collection of Tensors.
    """
    if hasattr(x, "to") and callable(x.to) and device:
        return x.to(device=device)
    return x


@move.register(dict)
def move_dict(x: Dict[K, V], device: Union[str, torch.device]) -> Dict[K, V]:
    return type(x)(**{
        move(k, device): move(v, device) for k, v in x.items()
    })


@move.register(list)
@move.register(tuple)
@move.register(set)
def move_sequence(x: Sequence[T], device: Union[str, torch.device]) -> Sequence[T]:
    if is_namedtuple(x):
        return type(x)(*[move(v, device) for v in x])
    return type(x)(move(v, device) for v in x)



@singledispatch
def move_(x: Any, device: Union[str, torch.device]) -> None:
    """Moves x *in-place* to the specified device if possible. else x is unchanged.

    Returns None.
    """
    if hasattr(x, "to_") and callable(x.to_) and device:
        x.to_(device=device)


from gym import spaces
from collections.abc import Mapping as MappingABC


@move_.register(dict)
@move_.register(spaces.Dict)
@move_.register(MappingABC)
def _move_dict_inplace(x: Dict[K, V], device: Union[str, torch.device]) -> None:
    for k, v in x.items():
        move_(v, device)


@move_.register(list)
@move_.register(tuple)
@move_.register(set)
def _move_sequence_inplace(x: Sequence[T], device: Union[str, torch.device]) -> None:
    for v in x:
        move_(v, device)
