""" Set of Utilities. """
import collections
import functools
import inspect
import itertools
import operator
import os
import random
import re
from collections import OrderedDict, defaultdict, deque
from collections.abc import MutableMapping
from dataclasses import Field, fields
from functools import reduce
from inspect import getsourcefile, isabstract, isclass
from itertools import filterfalse, groupby
from pathlib import Path
from typing import (Any, Callable, Deque, Dict, Iterable, List, MutableMapping,
                    Optional, Set, Tuple, Type, TypeVar, Union)

import numpy as np
import torch
from simple_parsing import field
from torch import Tensor, cuda, nn

cuda_available = cuda.is_available()
gpus_available = cuda.device_count()

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


def mean(values: Iterable[T]) -> T:
    values = list(values)
    return sum(values) / len(values)


def n_consecutive(items: Iterable[T], n: int=2, yield_last_batch=True) -> Iterable[Tuple[T, ...]]:
    values: List[T] = []
    for item in items:
        values.append(item)
        if len(values) == n:
            yield tuple(values)
            values.clear()
    if values and yield_last_batch:
        yield tuple(values)


def fix_channels(x_batch: Tensor) -> Tensor:
    # TODO: Move this to data_utils.py
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


def add_prefix(some_dict: Dict[str, T], prefix: str="", sep=" ") -> Dict[str, T]:
    """Adds the given prefix to all the keys in the dictionary that don't already start with it. 
    
    Parameters
    ----------
    - some_dict : Dict[str, T]
    
        Some dictionary.
    - prefix : str, optional, by default ""
    
        A string prefix to append.
    
    - sep : str, optional, by default " "

        A string separator to add between the `prefix` and the existing keys
        (which do no start by `prefix`). 

    
    Returns
    -------
    Dict[str, T]
        A new dictionary where all keys start with the prefix.


    Examples:
    -------
    >>> add_prefix({"a": 1}, prefix="bob", sep="")
    {'boba': 1}
    >>> add_prefix({"a": 1}, prefix="bob")
    {'bob a': 1}
    >>> add_prefix({"a": 1}, prefix="a")
    {'a': 1}
    >>> add_prefix({"a": 1}, prefix="a ")
    {'a': 1}
    >>> add_prefix({"a": 1}, prefix="a", sep="/")
    {'a': 1}
    """
    if not prefix:
        return some_dict
    result: Dict[str, T] = type(some_dict)()
    
    if sep and prefix.endswith(sep):
        prefix = prefix.rstrip(sep)

    for key, value in some_dict.items():
        new_key = key if key.startswith(prefix) else (prefix + sep + key)
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
    import random

    import numpy as np
    import torch
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def prod(iterable: Iterable[T]) -> T:
    """ Like sum() but returns the product of all numbers in the iterable.

    >>> prod(range(1, 5))
    24
    """
    return reduce(operator.mul, iterable, 1)


def to_optional_tensor(x: Optional[Union[Tensor, np.ndarray, List]]) -> Optional[Tensor]:
    """ Converts `x` into a Tensor if `x` is not None, else None. """
    return x if x is None else torch.as_tensor(x)


def common_fields(a, b) -> Iterable[Tuple[str, Tuple[Field, Field]]]:
    # If any attributes are common to both the Experiment and the State,
    # copy them over to the Experiment.
    a_fields = fields(a)
    b_fields = fields(b)
    for field_a in a_fields:
        name_a: str = field_a.name
        value_a = getattr(a, field_a.name) 
        for field_b in b_fields:
            name_b: str = field_b.name
            value_b = getattr(b, field_b.name)
            if name_a == name_b:
                yield name_a, (value_a, value_b)


def add_dicts(d1: Dict, d2: Dict, add_values=True) -> Dict:
    result = d1.copy()
    for key, v2 in d2.items():
        if key not in d1:
            result[key] = v2
        elif isinstance(v2, dict):
            result[key] = add_dicts(d1[key], v2, add_values=add_values)
        elif not add_values:
            result[key] = v2
        else:
            result[key] = d1[key] + v2
    return result


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

D = TypeVar("D", bound=Dict)

def flatten_dict(d: D, separator: str="/") -> D:
    """Flattens the given nested dict, adding `separator` between keys at different nesting levels.

    Args:
        d (Dict): A nested dictionary
        separator (str, optional): Separator to use. Defaults to "/".

    Returns:
        Dict: A flattened dictionary.
    """
    result = type(d)()
    for k, v in d.items():
        if isinstance(v, dict):
            for ki, vi in flatten_dict(v, separator=separator).items():
                key = f"{k}{separator}{ki}"
                result[key] = vi
        else:
            result[k] = v
    return result


def unique_consecutive(iterable: Iterable[T], key: Callable[[T], Any]=None) -> Iterable[T]:
    """List unique elements, preserving order. Remember only the element just seen.
    
    >>> list(unique_consecutive('AAAABBBCCDAABBB'))
    ['A', 'B', 'C', 'D', 'A', 'B']
    >>> list(unique_consecutive('ABBCcAD', str.lower))
    ['A', 'B', 'C', 'A', 'D']
    
    Recipe taken from itertools docs: https://docs.python.org/3/library/itertools.html
    """
    return map(next, map(operator.itemgetter(1), groupby(iterable, key)))


def roundrobin(*iterables: Iterable[T]) -> Iterable[T]:
    """
    roundrobin('ABC', 'D', 'EF') --> A D E B F C

    Recipe taken from itertools docs: https://docs.python.org/3/library/itertools.html
    """
    # Recipe credited to George Sakkis
    num_active = len(iterables)
    nexts = itertools.cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for next_ in nexts:
                yield next_()
        except StopIteration:
            # Remove the iterator we just exhausted from the cycle.
            num_active -= 1
            nexts = itertools.cycle(itertools.islice(nexts, num_active))


def take(iterable: Iterable[T], n: Optional[int]) -> Iterable[T]:
    """ Takes only the first `n` elements from `iterable`.
    
    if `n` is None, returns the entire iterable.
    """
    return itertools.islice(iterable, n) if n is not None else iterable


def camel_case(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    while "__" in s2:
        s2 = s2.replace("__", "_")
    return s2

def constant(v: T, **kwargs) -> T:
    return field(default=v, init=False, **kwargs)


def dict_union(*dicts: Dict[K, V], dict_factory=OrderedDict) -> Dict[K, V]:
    """ Simple dict union until we use python 3.9
    
    >>> from collections import OrderedDict
    >>> a = OrderedDict(a=1, b=2, c=3)
    >>> b = OrderedDict(c=5, d=6, e=7)
    >>> dict_union(a, b)
    OrderedDict([('a', 1), ('b', 2), ('c', 5), ('d', 6), ('e', 7)])
    >>> a = OrderedDict(a=1, b=OrderedDict(c=2, d=3))
    >>> b = OrderedDict(a=2, b=OrderedDict(c=3, e=6))
    >>> dict_union(a, b)
    OrderedDict([('a', 2), ('b', OrderedDict([('c', 3), ('d', 3), ('e', 6)]))])

    """
    result: Dict = dict_factory()
    if not dicts:
        return result
    assert len(dicts) >= 1
    all_keys: Set[str] = set()
    all_keys.update(*dicts)
    all_keys = sorted(all_keys)

    # Create a neat generator of generators.
    all_values: Iterable[Tuple[V, Iterable[K]]] = (
        (k, [d[k] for d in dicts if k in d]) for k in all_keys
    )
    for k, values in all_values:
        sub_dicts: List[Dict] = []
        for i, v in enumerate(values):
            if isinstance(v, dict):
                sub_dicts.append(v)
            else:
                new_value = v
        if len(sub_dicts) == (i + 1):
            # We only this here if all values for key `k` were dictionaries.
            new_value = dict_union(*sub_dicts, dict_factory=dict_factory)
        
        result[k] = new_value
    return result


K = TypeVar("K")
V = TypeVar("V")
M = TypeVar("M")

def zip_dicts(*dicts: Dict[K, V],
               missing: M = None) -> Iterable[Tuple[K, Tuple[Union[M, V], ...]]]:
    """Iterator over the union of all keys, giving the value from each dict if
    present, else `missing`.
    """
    # If any attributes are common to both the Experiment and the State,
    # copy them over to the Experiment.
    keys = set(itertools.chain(*dicts))
    for key in keys:
        yield (key, tuple(d.get(key, missing) for d in dicts))


def dict_intersection(*dicts: Dict[K, V]) -> Iterable[Tuple[K, Tuple[V, ...]]]:
    """Gives back an iterator over the keys and values common to all dicts. """
    dicts = [dict(d) for d in dicts]
    common_keys = set(dicts[0])
    for d in dicts:
        common_keys.intersection_update(d)
    for key in common_keys:
        yield (key, tuple(d[key] for d in dicts))


def try_get(d: Dict[K, V], *keys: K, default: V = None) -> Optional[V]:
    for k in keys:
        if k in d:
            return d[k]
    return default


def remove_suffix(s: str, suffix: str) -> str:
    """ Remove the suffix from string s if present.
    Doing this manually until we start using python 3.9.
    
    >>> remove_suffix("bob.com", ".com")
    'bob'
    >>> remove_suffix("Henrietta", "match")
    'Henrietta'
    """
    i = s.rfind(suffix)
    if i == -1:
        # return s if not found.
        return s
    return s[:i]


def remove_prefix(s: str, prefix: str) -> str:
    """ Remove the prefix from string s if present.
    Doing this manually until we start using python 3.9.
    
    >>> remove_prefix("bob.com", "bo")
    'b.com'
    >>> remove_prefix("Henrietta", "match")
    'Henrietta'
    """
    if not s.startswith(prefix):
        return s
    return s[len(prefix):]
    

def get_all_subclasses_of(cls: Type[T]) -> Iterable[Type[T]]:
    scope_dict: Dict = globals()
    for name, var in scope_dict.items():
        if isclass(var) and issubclass(var, cls):
            yield var

def get_all_concrete_subclasses_of(cls: Type[T]) -> Iterable[Type[T]]:
    yield from filterfalse(inspect.isabstract, get_all_subclasses_of(cls))


def get_path_to_source_file(cls: Type) -> Path:
    cwd = Path(os.getcwd())
    source_path = Path(getsourcefile(cls)).absolute()
    source_file = source_path.relative_to(cwd)
    return source_file


if __name__ == "__main__":
    import doctest
    doctest.testmod()
