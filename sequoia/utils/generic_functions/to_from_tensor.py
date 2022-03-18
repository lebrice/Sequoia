from functools import singledispatch
from typing import Any, Dict, Mapping, Optional, Tuple, TypeVar, Union

import numpy as np
import torch
from gym import Space, spaces
from torch import Tensor

T = TypeVar("T")


@singledispatch
def from_tensor(space: Space, sample: Union[Tensor, Any]) -> Union[np.ndarray, Any]:
    """Converts a Tensor into a sample from the given space."""
    if isinstance(sample, Tensor):
        return sample.cpu().numpy()
    return sample


@from_tensor.register
def _(space: spaces.Discrete, sample: Tensor) -> int:
    if isinstance(sample, Tensor):
        v = sample.item()
        int_v = int(v)
        if int_v != v:
            raise ValueError(f"Value {sample} isn't an integer, so it can't be from space {space}!")
        return int_v
    elif isinstance(sample, np.ndarray):
        assert sample.size == 1, sample
        return int(sample)
    return sample


@from_tensor.register
def _(
    space: spaces.Dict, sample: Dict[str, Union[Tensor, Any]]
) -> Dict[str, Union[np.ndarray, Any]]:
    return {key: from_tensor(space[key], value) for key, value in sample.items()}


from sequoia.utils.generic_functions._namedtuple import is_namedtuple


@from_tensor.register
def _(space: spaces.Tuple, sample: Tuple[Union[Tensor, Any]]) -> Tuple[Union[np.ndarray, Any]]:
    if not isinstance(sample, tuple):
        # BUG: Sometimes instead of having a sample of Tuple(Discrete(2))
        # be `(1,)`, its `array([1])` instead.
        sample = tuple(sample)
    values_gen = (from_tensor(space[i], value) for i, value in enumerate(sample))
    if is_namedtuple(sample):
        return type(sample)(*values_gen)
    return tuple(values_gen)


@singledispatch
def to_tensor(
    space: Space, sample: Union[np.ndarray, Any], device: torch.device = None
) -> Union[np.ndarray, Any]:
    """Converts a sample from the given space into a Tensor."""
    if sample is None:
        return sample
    return torch.as_tensor(sample, device=device)


@to_tensor.register
def _(
    space: spaces.MultiBinary, sample: np.ndarray, device: torch.device = None
) -> Dict[str, Union[Tensor, Any]]:
    return torch.as_tensor(sample, device=device, dtype=torch.bool)


@to_tensor.register
def _(
    space: spaces.Tuple,
    sample: Tuple[Union[np.ndarray, Any], ...],
    device: torch.device = None,
) -> Tuple[Union[Tensor, Any], ...]:
    if sample is None:
        assert all(isinstance(item_space, Sparse) for item_space in space.spaces)
        assert all(item_space.sparsity == 1.0 for item_space in space.spaces)
        # todo: What to do in this context?
        return None
        return np.full(
            [
                len(space.spaces),
            ],
            fill_value=None,
            dtype=np.object_,
        )
    if any(v is None for v in sample):
        assert False, (space, sample, device)
    return tuple(to_tensor(subspace, sample[i], device) for i, subspace in enumerate(space.spaces))


from typing import NamedTuple

from sequoia.common.spaces.named_tuple import NamedTupleSpace


@to_tensor.register
def _(space: NamedTupleSpace, sample: NamedTuple, device: torch.device = None):
    return space.dtype(
        **{
            key: to_tensor(space[i], sample[i], device=device)
            for i, key in enumerate(space._spaces.keys())
        }
    )


from sequoia.common.spaces.sparse import Sparse


@to_tensor.register(Sparse)
def sparse_sample_to_tensor(
    space: Sparse, sample: Union[Optional[Any], np.ndarray], device: torch.device = None
) -> Optional[Union[Tensor, np.ndarray]]:
    if space.sparsity == 1.0:
        if isinstance(space.base, spaces.MultiDiscrete):
            assert all(v == None for v in sample)
            return np.array([None if v == None else v for v in sample])
        if sample is not None:
            assert isinstance(sample, np.ndarray) and sample.dtype == np.object
            assert not sample.shape
        return None
    if space.sparsity == 0.0:
        # Do we need to convert dtypes here though?
        return to_tensor(space.base, sample, device)
    # 0 < sparsity < 1
    if isinstance(sample, np.ndarray) and sample.dtype == np.object:
        return np.array([None if v == None else v for v in sample])

    assert False, (space, sample)
