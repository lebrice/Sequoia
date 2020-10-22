""" TODO: Idea, create a new OptionalSpace class, (along with OptionalTuple,
OptionalDiscrete, etc) that adds has a probability of sampling `None` rather than
from the 'wrapped' space, and recognizes 'None' as being in the space. 
"""
from typing import Any, Dict, Generic, Optional, Tuple, TypeVar, Union

import gym
import numpy as np
from gym import spaces

T = TypeVar("T")


class Sparse(gym.Space, Generic[T]):
    def __init__(self, base: gym.Space, none_prob: float = 0.):
        self.base = base
        self.none_prob = none_prob
        # Would it ever cause a problem to have different dtypes for different
        # instances of the same space?
        # dtype = self.base.dtype if none_prob != 0. else np.object_ 
        super().__init__(shape=self.base.shape, dtype=np.object_)

    def seed(self, seed=None):
        super().seed(seed)
        return self.base.seed(seed=seed)

    def sample(self) -> Optional[T]:
        if self.none_prob == 0:
            return self.base.sample()
        if self.none_prob == 1.:
            return None
        p = self.np_random.random()
        if p <= self.none_prob:
            return None
        else:
            return self.base.sample()

    def contains(self, x: Optional[T]):
        """
        Return boolean specifying if x is a valid
        member of this space
        """
        return x is None or self.base.contains(x)

    def __repr__(self):
        return f"Sparse({self.base}, none_prob={self.none_prob})"
    
    def __eq__(self, other: Any):
        if not isinstance(other, Sparse):
            return NotImplemented
        return other.base == self.base and other.none_prob == self.none_prob


    def to_jsonable(self, sample_n):
        assert False, sample_n
        super().to_jsonable
        # serialize as dict-repr of vectors
        return {key: space.to_jsonable([sample[key] for sample in sample_n]) \
                for key, space in self.spaces.items()}

    def from_jsonable(self, sample_n):
        assert False, sample_n
        dict_of_list = {}
        for key, space in self.spaces.items():
            dict_of_list[key] = space.from_jsonable(sample_n[key])
        ret = []
        for i, _ in enumerate(dict_of_list[key]):
            entry = {}
            for key, value in dict_of_list.items():
                entry[key] = value[i]
            ret.append(entry)
        return ret

# from gym.spaces.utils import flatdim, flatten
from functools import singledispatch

import gym.spaces.utils
import gym.vector.utils
from gym.vector.utils import (batch_space, concatenate, create_empty_array,
                              create_shared_memory)

# Customize how these functions handle `Sparse` spaces by making them
# singledispatch callables and registering a new callable.

def _is_singledispatch(module_function):
    return hasattr(module_function, "registry")

if not _is_singledispatch(gym.spaces.utils.flatdim):
    gym.spaces.utils.flatdim = singledispatch(gym.spaces.utils.flatdim)

if not _is_singledispatch(gym.spaces.utils.flatten):
    gym.spaces.utils.flatten = singledispatch(gym.spaces.utils.flatten)

if not _is_singledispatch(gym.spaces.utils.unflatten):
    gym.spaces.utils.unflatten = singledispatch(gym.spaces.utils.unflatten)


@gym.spaces.utils.flatdim.register
def flatdim_sparse(space: Sparse) -> int:
    print(f"flat dim of sparse {space}: {gym.spaces.utils.flatdim(space.base)}")
    return gym.spaces.utils.flatdim(space.base)

@gym.spaces.utils.flatten.register(Sparse)
def flatten_sparse(space: Sparse[T], x: Optional[T]) -> Optional[np.ndarray]:
    return np.array([None]) if x is None else gym.spaces.utils.flatten(space.base, x)

@gym.spaces.utils.unflatten.register(Sparse)
def unflatten_sparse(space: Sparse[T], x: np.ndarray) -> Optional[T]:
    if len(x) == 1 and x[0] is None:
        return None
    else:
        return gym.spaces.utils.unflatten(space.base, x)

import multiprocessing as mp
from multiprocessing import Array, Value
from multiprocessing.context import BaseContext

import gym.vector.utils.shared_memory

if not _is_singledispatch(gym.vector.utils.shared_memory.create_shared_memory):
    gym.vector.utils.shared_memory.create_shared_memory = singledispatch(
        gym.vector.utils.shared_memory.create_shared_memory
    )

from gym.vector.utils.shared_memory import write_base_to_shared_memory
from ctypes import c_bool

@gym.vector.utils.shared_memory.create_shared_memory.register(Sparse)
def create_shared_memory_for_sparse_space(space: Sparse, n: int = 1, ctx: BaseContext = mp):
    # The shared memory should be something that can accomodate either 'None'
    # or a sample from the space. Therefore we should probably just create the
    # array for the base space, but then how would store a 'None' value in that
    # space?
    # What if we return a tuple or something, in which we actually add an 'is-none'
    print(f"Creating shared memory for {n} entries from space {space}")
    
    return {
        "is_none": ctx.Array(c_bool, np.zeros(n, dtype=np.bool)),
        "value": gym.vector.utils.shared_memory.create_shared_memory(space.base, n, ctx)
    }

# Writing to shared memory:

from gym.vector.utils.shared_memory import \
    write_to_shared_memory as write_to_shared_memory_


def write_to_shared_memory(index: int,
                           value: Optional[T],
                           shared_memory: Union[Dict, Tuple, BaseContext.Array],
                           space: Union[Sparse[T], gym.Space]):
    print(f"Writing entry from space {space} at index {index} in shared memory")
    if isinstance(space, Sparse):
        assert isinstance(shared_memory, dict)
        is_none_array = shared_memory["is_none"]
        value_array = shared_memory["value"]
        assert False, index
        assert False, is_none_array

        is_none_array[index] = value is None

        if value is not None:
            return write_to_shared_memory(index, value, value_array, space.base)
    else:
        # TODO: Would this cause a problem, say in the case where we have a
        # regular space like Tuple that contains some Sparse spaces, then would
        # calling this "old" function here prevent this "new" function from
        # being used on the children?
        return write_to_shared_memory_(index, value, shared_memory, space)

gym.vector.utils.shared_memory.write_to_shared_memory = write_to_shared_memory

# Reading from shared memory:

from gym.vector.utils.shared_memory import \
    read_from_shared_memory as read_from_shared_memory_

def read_from_shared_memory(shared_memory: Union[Dict, Tuple, BaseContext.Array],
                            space: Sparse,
                            n: int = 1):
    print(f"Reading {n} entries from space {space} from shared memory")
    if isinstance(space, Sparse):
        assert isinstance(shared_memory, dict)
        is_none_array = list(shared_memory["is_none"])
        value_array = shared_memory["value"]
        assert len(is_none_array) == len(value_array) == n
        
        # This might include some garbage (or default) values, which weren't
        # set.
        read_values = read_from_shared_memory(value_array, space.base, n)
        print(f"Read values from space: {read_values}")
        print(f"is_none array: {list(is_none_array)}")
        # assert False, (list(is_none_array), read_values, space)
        values = [
            None if is_none_array[index] else read_values[index]
            for index in range(n)
        ]
        print(f"resulting values: {values}")
        return values
        return read_from_shared_memory_(shared_memory, space.base, n)
    return read_from_shared_memory_(shared_memory, space, n)

gym.vector.utils.shared_memory.read_from_shared_memory = read_from_shared_memory


assert _is_singledispatch(gym.spaces.utils.flatdim)

# These two aren't causing problems as they are.
if not _is_singledispatch(gym.vector.utils.batch_space):
    gym.vector.utils.batch_space = singledispatch(gym.vector.utils.batch_space)

if not _is_singledispatch(gym.vector.utils.concatenate):
    gym.vector.utils.concatenate = singledispatch(gym.vector.utils.concatenate)


# if not hasattr(gym.vector.utils.concatenate, "registry"):
#     concatenate = singledispatch(concatenate)
#     gym.vector.utils.concatenate = concatenate


@gym.vector.utils.batch_space.register
def batch_sparse_spaces(space: Sparse, n: int=1) -> gym.Space:
    assert False, f"WOOT WOOT!: {space}, {n}"

@gym.vector.utils.concatenate.register
def concatenate_sparse_spaces(space: Sparse, n: int=1) -> gym.Space:
    assert False, "WOOT WOOT!!"

