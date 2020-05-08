""" Set of Utilities. """

import collections
import functools
from collections import OrderedDict, defaultdict, deque
from collections.abc import MutableMapping
from pathlib import Path
from typing import (
    Any, Deque, Dict, Iterable, List, MutableMapping, Optional, Set, Tuple,
    TypeVar, Union)

import numpy as np
import torch
from torch import Tensor, nn

cuda_available = torch.cuda.is_available()
gpus_available = torch.cuda.device_count()

T = TypeVar("T")


def n_consecutive(items: Iterable[T], n: int=2, yield_last_batch=True) -> Iterable[Tuple[T, ...]]:
    values: List[T] = []
    for item in items:
        values.append(item)
        if len(values) == n:
            yield tuple(values)
            values.clear()
    if values and yield_last_batch:
        yield tuple(values)


def to_list(tensors: Iterable[Union[Any, Tensor]]) -> List[float]:
    """Converts a list of tensors into a list of values.
    
    `tensors` must contain scalar tensors.
    
    Parameters
    ----------
    - tensors : Iterable[Union[T, Tensor]]
    
        some scalar tensors
    
    Returns
    -------
    List[float]
        A list of their values.
    """
    if tensors is None:
        return []
    
    return list(map(
        lambda v: (v.item() if isinstance(v, Tensor) else v),
        tensors,
        )
    )


def fix_channels(x_batch: Tensor) -> Tensor:
    if x_batch.dim() == 3:
        return x_batch.unsqueeze(1)
    else:
        if x_batch.shape[1] != min(x_batch.shape[1:]):
            return x_batch.transpose(1, -1)
        else:
            return x_batch


def to_dict_of_lists(list_of_dicts: Iterable[Dict[str, Any]]) -> Dict[str, List[Tensor]]:
    """ Returns a dict of lists given a list of dicts.
    
    Assumes that all dictionaries have the same keys as the first dictionary.
    
    Args:
        list_of_dicts (Iterable[Dict[str, Any]]): An iterable of dicts.
    
    Returns:
        Dict[str, List[Tensor]]: A Dict of lists.
    """
    result: Dict[str, List[Any]] = defaultdict(list)
    for i, d in enumerate(list_of_dicts):
        for key, value in d.items():
            result[key].append(value)
        assert d.keys() == result.keys(), f"Dict {d} at index {i} does not contain all the keys!"
    return result


def add_prefix(some_dict: Dict[str, T], prefix: str="") -> Dict[str, T]:
    """Adds the given prefix to all the keys in the dictionary that don't already start with it. 
    
    Parameters
    ----------
    - some_dict : Dict[str, T]
    
        Some dictionary.
    - prefix : str, optional, by default ""
    
        A string prefix to append.
    
    Returns
    -------
    Dict[str, T]
        A new dictionary where all keys start with the prefix.
    """
    if not prefix:
        return OrderedDict(some_dict.items())
    result: Dict[str, T] = OrderedDict()
    for key, value in some_dict.items():
        new_key = key if key.startswith(prefix) else (prefix + key)
        result[new_key] = value
    return result


def loss_str(loss_tensor: Tensor) -> str:
    loss = loss_tensor.item()
    if loss == 0:
        return "0"
    elif abs(loss) < 1e-3 or abs(loss) > 1e3:
        return f"{loss:.1e}"
    else:
        return f"{loss:.3f}"


def set_seed(seed: int):
    """ Set the pytorch/numpy random seed. """
    torch.manual_seed(seed)
    np.random.seed(seed)


def rsetattr(obj: Any, attr: str, val: Any) -> None:
    """ Taken from https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties """
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

# using wonder's beautiful simplification: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427

def rgetattr(obj: Any, attr: str, *args):
    """ Taken from https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties """
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def is_nonempty_dir(path: Path) -> bool:
    return path.is_dir() and len(list(path.iterdir())) > 0


if __name__ == "__main__":
    import doctest
    doctest.testmod()
