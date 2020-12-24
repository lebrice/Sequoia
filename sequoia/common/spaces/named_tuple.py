""" IDEA: Subclass of `gym.spaces.Tuple` that yields namedtuples,
as a bit of a hybrid between `gym.spaces.Dict` and `gym.spaces.Tuple`.
"""
import gym
from gym import Space, spaces
import numpy as np
from sequoia.utils import NamedTuple
from typing import Union, Sequence, Dict, Mapping, Type
from collections.abc import Mapping as MappingABC
from collections import namedtuple


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
            assert len(names) == len(spaces), "need to pass a name for each space"
            self._spaces = dict(zip(names, spaces))
        

        # NOTE: dict.values() is ordered since python 3.7.
        spaces = tuple(self._spaces.values())
        super().__init__(spaces)
        self.names: Sequence[str] = self._spaces.keys()
        self.dtype = dtype or namedtuple("NamedTuple", self.names)
        # idea: could use this _name attribute
        self._name = self.dtype.__name__
        
        # # IDEA: to make it simpler to recover NamedTupleSpaces after operations
        # # are performed ?
        # for name, space in self._spaces.items():
        #     space.__name = name

    
    def __getitem__(self, index: Union[int, str]) -> Space:
        if isinstance(index, str):
            return self._spaces[index]
        return super().__getitem__(index)


    def __repr__(self):
        # TODO: Tricky: decide what name to show for the space class:
        cls_name = type(self).__name__
        # cls_name = self._name or type(self).__name__
        return f"{cls_name}(" + ", ".join([str(k) + "=" + str(s) for k, s in self._spaces.items()]) + ")"


    # def __repr__(self):
    #     return "Tuple(" + ", ". join([str(s) for s in self.spaces]) + ")"

    def sample(self):
        return self.dtype(*super().sample())

    def contains(self, x) -> bool:
        # TODO: Should we accept dataclasses as valid namedtuple space items?
        if isinstance(x, MappingABC):
            x = tuple(x.values())
        return super().contains(x)
        # if isinstance(x, list):
        #     x = tuple(x)  # Promote list to tuple for contains check
        # return isinstance(x, tuple) and len(x) == len(self.spaces) and all(
        #     space.contains(part) for (space,part) in zip(self.spaces,x))


from gym.vector.utils import batch_space

@batch_space.register(NamedTupleSpace)
def batch_namedtuple_space(space: NamedTupleSpace, n: int = 1):
    assert False, "HEYO"
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
