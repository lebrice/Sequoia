from .sparse_space import Sparse

import gym
from gym import spaces
import numpy as np
import pytest


base_spaces = [
    spaces.Discrete(n=10),
    spaces.Box(0,1, [3, 32, 32], dtype=np.float32),
    spaces.Tuple([
        spaces.Discrete(n=10),
        spaces.Box(0,1, [3, 32, 32], dtype=np.float32),
    ]),
    spaces.Dict({
        "x": spaces.Tuple([
            spaces.Discrete(n=10),
            spaces.Box(0,1, [3, 32, 32], dtype=np.float32),
        ]),
        "t": spaces.Discrete(1),
    }),
]


@pytest.mark.parametrize("base_space", base_spaces)
def test_sample(base_space: gym.Space):
    space = Sparse(base_space, none_prob=0.)
    samples = [space.sample() for i in range(100)]
    assert all(sample is not None for sample in samples)
    assert all(sample in base_space for sample in samples)

    space = Sparse(base_space, none_prob=0.5)
    samples = [space.sample() for i in range(100)]
    assert not all([sample is None for sample in samples])
    assert not all([sample is not None for sample in samples])
    assert all([sample in base_space for sample in samples if sample is not None])

    space = Sparse(base_space, none_prob=1.0)
    samples = [space.sample() for i in range(100)]
    assert all(sample is None for sample in samples)

@pytest.mark.parametrize("none_prob", [0., 0.5, 1.])
@pytest.mark.parametrize("base_space", base_spaces)
def test_contains(base_space: gym.Space, none_prob: float):
    space = Sparse(base_space, none_prob=none_prob)
    samples = [space.sample() for i in range(100)]
    assert all(sample in space for sample in samples)

from gym.vector.utils import batch_space, concatenate


@pytest.mark.parametrize("base_space", base_spaces)
def test_batching_works(base_space: gym.Space, n: int = 3):
    batched_base_space = batch_space(base_space, n)
    sparse_space = Sparse(base_space)

    batched_sparse_space = batch_space(sparse_space, n)
    
    
    base_batch = batched_base_space.sample()
    sparse_batch = batched_sparse_space.sample()
    assert len(base_batch) == len(sparse_batch)


@pytest.mark.parametrize("base_space", base_spaces)
def test_batching_works(base_space: gym.Space, n: int = 3):
    batched_base_space = batch_space(base_space, n)
    sparse_space = Sparse(base_space)

    batched_sparse_space = batch_space(sparse_space, n)
    base_batch = batched_base_space.sample()
    sparse_batch = batched_sparse_space.sample()


def test_change_persists_after_import():    
    from gym.vector.utils import concatenate
    from .sparse_space import Sparse
    assert hasattr(gym.vector.utils.concatenate, "registry")

from gym.spaces.utils import flatten_space, flatdim, flatten


@pytest.mark.parametrize("base_space", base_spaces)
def test_flatdim(base_space: gym.Space):
    sparse_space = Sparse(base_space, none_prob=0.)

    base_flat_dims = flatdim(base_space)
    sparse_flat_dims = flatdim(sparse_space)

    assert base_flat_dims == sparse_flat_dims


@pytest.mark.parametrize("base_space", base_spaces)
def test_flatdim(base_space: gym.Space):
    sparse_space = Sparse(base_space, none_prob=0.)

    base_flat_dims = flatdim(base_space)
    sparse_flat_dims = flatdim(sparse_space)
    assert base_flat_dims == sparse_flat_dims
    
    # The flattened dimensions shouldn't depend on the none_prob.
    sparse_space = Sparse(base_space, none_prob=1.)
    sparse_flat_dims = flatdim(sparse_space)
    assert base_flat_dims == sparse_flat_dims

def equals(value, expected) -> bool:
    assert type(value) == type(expected)
    if isinstance(value, (int, float, bool)):
        return value == expected
    if isinstance(value, np.ndarray):
        return value.tolist() == expected.tolist()
    if isinstance(value, (tuple, list)):
        assert len(value) == len(expected)
        return all(equals(a_v, e_v) for a_v, e_v in zip(value, expected))
    if isinstance(value, dict):
        assert len(value) == len(expected)
        for k in expected.keys():
            if k not in value:
                return False
            if not equals(value[k], expected[k]):
                return False
        return True
    return value == expected
        
    


@pytest.mark.parametrize("base_space", base_spaces)
def test_seeding_works(base_space: gym.Space):
    sparse_space = Sparse(base_space, none_prob=0.)

    base_space.seed(123)
    base_sample = base_space.sample()
    
    sparse_space.seed(123)
    sparse_sample = sparse_space.sample()

    assert equals(base_sample, sparse_sample)


@pytest.mark.parametrize("base_space", base_spaces)
def test_flatten(base_space: gym.Space):
    sparse_space = Sparse(base_space, none_prob=0.)
    base_space.seed(123)
    base_sample = base_space.sample()
    flattened_base_sample = flatten(base_space, base_sample)
    
    sparse_space.seed(123)
    sparse_sample = sparse_space.sample()
    flattened_sparse_sample = flatten(sparse_space, sparse_sample)
    
    assert equals(flattened_base_sample, flattened_sparse_sample)