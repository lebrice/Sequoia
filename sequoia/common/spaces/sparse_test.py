from .sparse import Sparse
from typing import Iterable

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


def is_sparse(iterable: Iterable[bool]) -> bool:
    """ Returns wether some (but not all) values in the iterable are None. """
    none_values: int = 0
    non_none_values: int = 0
    for value in iterable:
        if value is None:
            none_values += 1
            if non_none_values:
                return True
        else:
            non_none_values += 1
            if none_values:
                return True
    return False
    # Equivalent, but with a copy:
    values = list(values)
    return any(v is None for v in values) and not all(v is None for v in values)


@pytest.mark.parametrize("base_space", base_spaces)
def test_sample(base_space: gym.Space):
    space = Sparse(base_space, sparsity=0.)
    samples = [space.sample() for i in range(100)]
    assert all(sample is not None for sample in samples)
    assert all(sample in base_space for sample in samples)

    space = Sparse(base_space, sparsity=0.5)
    samples = [space.sample() for i in range(100)]
    assert is_sparse(samples)
    assert all([sample in base_space for sample in samples if sample is not None])

    space = Sparse(base_space, sparsity=1.0)
    samples = [space.sample() for i in range(100)]
    assert all(sample is None for sample in samples)


@pytest.mark.parametrize("sparsity", [0., 0.5, 1.])
@pytest.mark.parametrize("base_space", base_spaces)
def test_contains(base_space: gym.Space, sparsity: float):
    space = Sparse(base_space, sparsity=sparsity)
    samples = [space.sample() for i in range(100)]
    assert all(sample in space for sample in samples)


from sequoia.common.spaces.utils import batch_space, concatenate


@pytest.mark.parametrize("base_space", base_spaces)
def test_batching_works(base_space: gym.Space, n: int = 3):
    batched_base_space = batch_space(base_space, n)
    sparse_space = Sparse(base_space)

    batched_sparse_space = batch_space(sparse_space, n)
    
    
    base_batch = batched_base_space.sample()
    sparse_batch = batched_sparse_space.sample()
    assert len(base_batch) == len(sparse_batch)

# @pytest.mark.xfail(reason="TODO: Need to decide how we want the sparsity to "
#                           "affect the batching of Tuple or Dict spaces.")
@pytest.mark.parametrize("base_space", base_spaces)
@pytest.mark.parametrize("sparsity", [0., 0.5, 1.0])
def test_batching_works(base_space: gym.Space, sparsity: float, n: int = 10):
    batched_base_space = batch_space(base_space, n)
    
    sparse_space = Sparse(base_space, sparsity=sparsity)
    batched_sparse_space = batch_space(sparse_space, n)
    
    batched_base_space.seed(123)
    base_batch = batched_base_space.sample()
    
    batched_sparse_space.seed(123)
    sparse_batch = batched_sparse_space.sample()

    if sparsity == 0:
        # When there is no sparsity, the batching is the same as batching the
        # same space.
        assert equals(base_batch, sparse_batch)
    elif sparsity == 1:
        assert sparse_batch is None
        # assert len(sparse_batch) == n
        # assert sparse_batch == tuple([None] * n)
    else:
        assert len(sparse_batch) == n
        assert isinstance(sparse_batch, tuple)

        for i, value in enumerate(sparse_batch):
            if value is not None:
                assert value in base_space

        # There should be some sparsity.
        assert (any(v is None for v in sparse_batch) and not
                all(v is None for v in sparse_batch)), sparse_batch


from gym.spaces.utils import flatten_space, flatdim, flatten

@pytest.mark.xfail(reason="When using the normal gym repo rather than the "
                          "fork, the change doesn't persist through an import.")
def test_change_doesnt_persist_after_import():
    """ When re-importing the `concatenate` function from `gym.vector.utils`,
    the changes aren't preserved.
    """
    from gym.vector.utils import concatenate
    from .sparse import Sparse
    assert hasattr(gym.vector.utils.numpy_utils.concatenate, "registry")
    assert hasattr(gym.vector.utils.batch_space, "registry")


def test_change_persists_after_full_import():
    """ When re-importing the `concatenate` function from
    `gym.vector.utils.numpy_utils`, the changes are preserved.
    """
    from sequoia.common.spaces.utils import concatenate
    from .sparse import Sparse
    assert hasattr(gym.vector.utils.numpy_utils.concatenate, "registry")
    assert hasattr(gym.vector.utils.batch_space, "registry")



@pytest.mark.parametrize("base_space", base_spaces)
def test_flatdim(base_space: gym.Space):
    sparse_space = Sparse(base_space, sparsity=0.)

    base_flat_dims = flatdim(base_space)
    sparse_flat_dims = flatdim(sparse_space)

    assert base_flat_dims == sparse_flat_dims


@pytest.mark.parametrize("base_space", base_spaces)
def test_flatdim(base_space: gym.Space):
    sparse_space = Sparse(base_space, sparsity=0.)

    base_flat_dims = flatdim(base_space)
    sparse_flat_dims = flatdim(sparse_space)
    assert base_flat_dims == sparse_flat_dims
    
    # The flattened dimensions shouldn't depend on the sparsity.
    sparse_space = Sparse(base_space, sparsity=1.)
    sparse_flat_dims = flatdim(sparse_space)
    assert base_flat_dims == sparse_flat_dims


@pytest.mark.parametrize("base_space", base_spaces)
def test_seeding_works(base_space: gym.Space):
    sparse_space = Sparse(base_space, sparsity=0.)

    base_space.seed(123)
    base_sample = base_space.sample()
    
    sparse_space.seed(123)
    sparse_sample = sparse_space.sample()

    assert equals(base_sample, sparse_sample)


@pytest.mark.parametrize("base_space", base_spaces)
def test_flatten(base_space: gym.Space):
    sparse_space = Sparse(base_space, sparsity=0.)
    base_space.seed(123)
    base_sample = base_space.sample()
    flattened_base_sample = flatten(base_space, base_sample)
    
    sparse_space.seed(123)
    sparse_sample = sparse_space.sample()
    flattened_sparse_sample = flatten(sparse_space, sparse_sample)
    
    assert equals(flattened_base_sample, flattened_sparse_sample)

@pytest.mark.parametrize("base_space", base_spaces)
def test_equality(base_space: gym.Space):
    sparse_space = Sparse(base_space, sparsity=0.)
    other_space = Sparse(base_space, sparsity=0.)    
    assert sparse_space == other_space

    sparse_space = Sparse(base_space, sparsity=0.2)
    assert sparse_space != other_space
    
    sparse_space = Sparse(spaces.Tuple([base_space, base_space]), sparsity=0.)
    assert sparse_space != other_space
