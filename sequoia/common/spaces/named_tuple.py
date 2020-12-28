""" IDEA: Subclass of `gym.spaces.Tuple` that yields namedtuples,
as a bit of a hybrid between `gym.spaces.Dict` and `gym.spaces.Tuple`.
"""
from collections import namedtuple
from collections.abc import Mapping as MappingABC
from typing import Any, Dict, Mapping, Sequence, Tuple, Type, Union

import gym
import numpy as np
from gym import Space, spaces
from sequoia.utils import NamedTuple


class NamedTupleSpace(spaces.Tuple):
    """
    A tuple (i.e., product) of simpler spaces, with namedtuple samples.

    Example usage:
    
    ```python 
    self.observation_space = NamedTupleSpace(x=spaces.Discrete(2), t=spaces.Discrete(3))
    ```

    Note: here the dtype is actually the type of namedtuple to use, not a
    numpy dtype.
    """
    def __init__(self,
                 spaces: Union[Mapping[str, Space], Sequence[Space]] = None,
                 names: Sequence[str] = None,
                 dtype: Type[NamedTuple] = None,
                 **kwargs):
        self._spaces: Dict[str, Space]
        if isinstance(spaces, MappingABC):
            assert names is None
            self._spaces = dict(spaces.items())
        elif kwargs:
            assert all(isinstance(k, str) and isinstance(v, Space)
                       for k, v in kwargs.items())
            self._spaces = kwargs
        else:
            # if not names:
            #     try:
            #         names = [getattr(space, "__name") for space in spaces]
            #     except AttributeError:
            #         pass
            assert names is not None, "need to pass names when spaces isn't a mapping."
            assert spaces and len(names) == len(spaces), "need to pass a name for each space"
            self._spaces = dict(zip(names, spaces))

        # NOTE: dict.values() is ordered since python 3.7.
        spaces = tuple(self._spaces.values())
        super().__init__(spaces)
        self.names: Sequence[str] = tuple(self._spaces.keys())
        self.dtype: Type[Tuple] = dtype or namedtuple("NamedTuple", self.names)
        # idea: could use this _name attribute to change the __repr__ first part
        self._name = self.dtype.__name__
    
    def __getitem__(self, index: Union[int, str]) -> Space:
        if isinstance(index, str):
            return self._spaces[index]
        return super().__getitem__(index)

    def __repr__(self):
        # TODO: Tricky: decide what name to show for the space class:
        cls_name = type(self).__name__
        # cls_name = self._name or type(self).__name__
        return f"{cls_name}(" + ", ".join([
            str(k) + "=" + str(s) for k, s in self._spaces.items()
        ]) + ")"

    def __eq__(self, other: Union["NamedTupleSpace", Any]) -> bool:
        return isinstance(other, spaces.Tuple) and tuple(self.spaces) == tuple(other.spaces)

    def sample(self):
        return self.dtype(*super().sample())

    def contains(self, x) -> bool:
        # TODO: Should we accept dataclasses as valid namedtuple space items?
        if isinstance(x, MappingABC):
            x = tuple(x.values())
        return super().contains(x)


# See https://github.com/openai/gym/issues/2140 : Fix __eq__ of gym.spaces.Tuple
def __eq__(self, other: Union["NamedTupleSpace", Any]) -> bool:
    # BUG in openai gym: spaces passed to the spaces.Tuple constructor could
    # be a list of spaces, rather than a tuple, and so this might return
    # False when it shouldn't.
    return isinstance(other, spaces.Tuple) and tuple(self.spaces) == tuple(other.spaces)
spaces.Tuple.__eq__ = __eq__


from gym.vector.utils import batch_space


@batch_space.register(NamedTupleSpace)
def batch_namedtuple_space(space: NamedTupleSpace, n: int = 1):
    return NamedTupleSpace(spaces={
        key: batch_space(value, n) for key, value in space._spaces.items()
    }, dtype=space.dtype)

import torch
from sequoia.utils.generic_functions.to_from_tensor import to_tensor


@to_tensor.register(NamedTupleSpace)
def _to_tensor(space: NamedTupleSpace, sample: NamedTuple, device: torch.device = None):
    return type(sample)(**{
        key: to_tensor(space[i], sample[i], device=device) for i, key in enumerate(space._spaces.keys())
    })
