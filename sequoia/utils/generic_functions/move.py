"""Defines a singledispatch function to move objects to a given device.
"""
from functools import singledispatch
from typing import Dict, Sequence, TypeVar, Union

import torch

from sequoia.utils.generic_functions._namedtuple import is_namedtuple

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
    return type(x)(**{move(k, device): move(v, device) for k, v in x.items()})


@move.register(list)
@move.register(tuple)
@move.register(set)
def move_sequence(x: Sequence[T], device: Union[str, torch.device]) -> Sequence[T]:
    if is_namedtuple(x):
        return type(x)(*[move(v, device) for v in x])
    return type(x)(move(v, device) for v in x)
