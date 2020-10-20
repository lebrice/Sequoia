""" TODO: Idea, create a new OptionalSpace class, (along with OptionalTuple,
OptionalDiscrete, etc) that adds has a probability of sampling `None` rather than
from the 'wrapped' space, and recognizes 'None' as being in the space. 
"""
import gym
from gym import spaces
import numpy as np
from typing import Generic, TypeVar, Optional, Any

T = TypeVar("T")


class Sparse(gym.Space, Generic[T]):
    def __init__(self, base: gym.Space, none_prob: float = 0.):
        self.base = base
        self.none_prob = none_prob
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

import gym.vector.utils
from gym.vector.utils import batch_space, concatenate, create_empty_array, create_shared_memory, create_empty_array

import gym.spaces.utils
# from gym.spaces.utils import flatdim, flatten
from functools import singledispatch

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
def flatten_sparse(space: Sparse[T], x: Optional[np.ndarray]) -> Optional[T]:
    if len(x) == 1 and x[0] is None:
        return None
    else:
        return gym.spaces.utils.unflatten(space.base, x)

assert _is_singledispatch(gym.spaces.utils.flatdim)

# These two aren't causing problems as they are.
if not _is_singledispatch(gym.vector.utils.batch_space):
    gym.vector.utils.batch_space = singledispatch(gym.vector.utils.batch_space)

if not _is_singledispatch(gym.vector.utils.concatenate):
    gym.vector.utils.concatenate = singledispatch(gym.vector.utils.concatenate)


# if not hasattr(gym.vector.utils.concatenate, "registry"):
#     concatenate = singledispatch(concatenate)
#     gym.vector.utils.concatenate = concatenate


# @batch_space.register
# def batch_sparse_spaces(space: Sparse, n: int=1) -> gym.Space:
#     assert False, "WOOT WOOT!!"

# @concatenate.register
# def concatenate_sparse_spaces(space: Sparse, n: int=1) -> gym.Space:
#     assert False, "WOOT WOOT!!"

