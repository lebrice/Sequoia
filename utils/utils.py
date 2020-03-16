""" Set of Utilities. """

import collections
from collections import defaultdict, deque
from collections.abc import MutableMapping
from typing import (Any, Deque, Dict, Iterable, List, MutableMapping, Optional,
                    Tuple, TypeVar, Union)

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
    
    `tensots` must contain scalar tensors.Any
    
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


def to_dict_of_lists(list_of_dicts: List[Dict[str, Tensor]]) -> Dict[str, List[Tensor]]:
    # TODO: we have a list of dicts, change it into a dict of lists.
    result: Dict[str, List[Any]] = defaultdict(list)
    for i, d in enumerate(list_of_dicts):
        for key, tensor in d.items():
            result[key].append(tensor.cpu())
        assert d.keys() == result.keys()
    return result


def add_prefix(some_dict: Dict[str, T], prefix: str="") -> Dict[str, T]:
    return {prefix + key: value for key, value in some_dict.items()}


def loss_str(loss_tensor: Tensor) -> str:
    loss = loss_tensor.item()
    if loss == 0:
        return "0"
    elif abs(loss) < 1e-3 or abs(loss) > 1e3:
        return f"{loss:.1e}"
    else:
        return f"{loss:.3f}"

import functools

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
  

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    # cache = TensorCache(5)


    # d = TensorCache(5)
    # zero = torch.zeros(3,3)
    # d[zero] = torch.Tensor(123)

    # one = torch.ones(3,3)
    # batch = torch.stack([zero, one])
    # print("zero is in cache:", zero in d)
    # print("ones is in cache:", one in d)
    # print(torch.zeros(3,3) in d)
    # print(d[torch.zeros(3,3)])
