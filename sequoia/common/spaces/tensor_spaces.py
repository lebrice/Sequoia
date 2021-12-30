""" TODO: Maybe create a typed version of 'add_tensor_support' of gym_wrappers.convert_tensors
"""
from abc import ABC, abstractmethod
from contextlib import contextmanager
from inspect import isclass
from typing import Any, Optional, Sequence, TypeVar, Union

import gym
from gym.spaces.utils import flatten_space
import numpy as np
import torch
from gym import spaces
from .utils import batch_space
from torch import Tensor

# Dict of NumPy dtype -> torch dtype (when the correspondence exists)
numpy_to_torch_dtypes = {
    bool: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}
# Dict of torch dtype -> NumPy dtype
torch_to_numpy_dtypes = {value: key for (key, value) in numpy_to_torch_dtypes.items()}
T = TypeVar("T")


def get_numpy_dtype_equivalent_to(torch_dtype: torch.dtype) -> np.dtype:
    """TODO: Gets the numpy dtype equivalent to the given torch dtype."""

    def dtypes_equal(a: torch.dtype, b: torch.dtype) -> bool:
        return a == b  # simple for now.

    matching_dtypes = [v for k, v in torch_to_numpy_dtypes.items() if dtypes_equal(k, torch_dtype)]
    if len(matching_dtypes) == 0:
        raise RuntimeError(f"Unable to find a numpy dtype equivalent to {torch_dtype}")
    if len(matching_dtypes) > 1:
        raise RuntimeError(f"Found more than one match for dtype {torch_dtype}: {matching_dtypes}")
    return np.dtype(matching_dtypes[0])


def get_torch_dtype_equivalent_to(numpy_dtype: np.dtype) -> torch.dtype:
    """TODO: Gets the torch dtype equivalent to the given np dtype."""

    def dtypes_equal(a: torch.dtype, b: torch.dtype) -> bool:
        return a == b  # simple for now.

    matching_dtypes = [v for k, v in numpy_to_torch_dtypes.items() if dtypes_equal(k, numpy_dtype)]
    if len(matching_dtypes) == 0:
        raise RuntimeError(f"Unable to find a torch dtype equivalent to {numpy_dtype}")
    if len(matching_dtypes) > 1:
        raise RuntimeError(f"Found more than one match for dtype {numpy_dtype}: {matching_dtypes}")
    return matching_dtypes[0]


def is_numpy_dtype(dtype: Any) -> bool:
    return isinstance(dtype, np.dtype) or isclass(dtype) and issubclass(dtype, np.generic)


def is_torch_dtype(dtype: Any) -> bool:
    return isinstance(dtype, torch.dtype)


def supports_tensors(space: gym.Space) -> bool:
    raise NotImplementedError(f"TODO: Create a generic function for this.")
    return isinstance(space, TensorSpace)
    pass


class TensorSpace(gym.Space, ABC):
    """Mixin class that makes a Space's `contains` and `sample` methods accept and
    produce tensors, respectively.
    """

    def __init__(self, *args, device: torch.device = None, **kwargs):
        # super().__init__(*args, **kwargs)

        self.device: Optional[torch.device] = torch.device(device) if device else None
        # Depending on the value passed to `dtype`
        dtype = kwargs.get("dtype")
        if dtype is None:
            if isinstance(self, (spaces.Discrete, spaces.MultiDiscrete)):
                # NOTE: They dont actually give a 'dtype' argument for these.
                self._numpy_dtype = np.dtype(np.int64)
                self._torch_dtype = torch.int64
            else:
                raise NotImplementedError(f"Space {self} doesn't have a `dtype`?")
        elif is_numpy_dtype(dtype):
            self._numpy_dtype = np.dtype(dtype)
            self._torch_dtype = get_torch_dtype_equivalent_to(dtype)
        elif is_torch_dtype(dtype):
            self._numpy_dtype = get_numpy_dtype_equivalent_to(dtype)
            self._torch_dtype = dtype
        elif str(dtype) == "float32":
            self._numpy_dtype = np.dtype(np.float32)
            self._torch_dtype = torch.float32
        else:
            assert not any(dtype == k for k in numpy_to_torch_dtypes)
            assert not any(dtype == k for k in torch_to_numpy_dtypes)
            raise NotImplementedError(f"Unsupported dtype {dtype} (of type {type(dtype)})")
        if "dtype" in kwargs:
            kwargs["dtype"] = self._numpy_dtype

        super().__init__(*args, **kwargs)
        self.dtype: torch.dtype = self._torch_dtype

    @contextmanager
    def using_np_dtype(self):
        """ Context manager that temporarily changes the attributes to the numpy equivalents. """
        self.dtype = self._numpy_dtype
        yield
        self.dtype = self._torch_dtype

    # TODO: Add these in.
    # @abstractmethod
    # def to(self: T, device: Union[str, torch.device]) -> T:
    #     """ Returns a new space, where the samples will be on device `device`. """

    # @abstractmethod
    # def to_(self, device: Union[str, torch.device]) -> None:
    #     """ Modifies this space in-place, so the the samples will be on device `device`. """

from sequoia.utils.generic_functions.to_from_tensor import from_tensor


@from_tensor.register(TensorSpace)
def from_tensor(space: TensorSpace, sample: Union[Tensor, Any]) -> Union[Tensor, Any]:
    # Do nothing: The samples from this space are supposed to be tensors.
    return sample


from .utils import create_shared_memory, create_empty_array

@create_shared_memory.register(TensorSpace)
def _create_tensorspace_shared_memory(space: TensorSpace, *args, **kwargs):
    with space.using_np_dtype():
        non_tensor_base_class = next(
            base_class for base_class in type(space).mro() if issubclass(base_class, gym.Space) and not issubclass(base_class, TensorSpace)
        )
        return create_shared_memory.dispatch(non_tensor_base_class)(space, *args, **kwargs)


@create_empty_array.register(TensorSpace)
def _create_tensor_empty_array(space: TensorSpace, n: int = 1, fn=np.zeros):
    non_tensor_base_class = next(
        base_class for base_class in type(space).mro() if issubclass(base_class, gym.Space) and
        not issubclass(base_class, TensorSpace)
    )
    mapping = {
        np.zeros: torch.zeros,
        np.ones: torch.ones,
    }
    from sequoia.utils.generic_functions import to_tensor
    
    if fn not in mapping:
        def torch_fn(*args, **kwargs):
            v = fn(*args, **kwargs)
            return to_tensor(v, device=space.device)
    else:
        torch_fn = mapping[fn]
    return create_empty_array.dispatch(non_tensor_base_class)(space, n=n, fn=torch_fn)

    # # with space.using_np_dtype():
    # from sequoia.utils.generic_functions.to_from_tensor import to_tensor
    # return to_tensor(np_empty_array, device=space.device)



class TensorBox(TensorSpace, spaces.Box):
    """Box space that accepts both Tensor and ndarrays."""

    def __init__(self, low, high, shape=None, dtype=np.float32, device: torch.device = None):
        super().__init__(low, high, shape=shape, dtype=dtype, device=device)
        self.low_tensor = torch.as_tensor(self.low, device=self.device)
        self.high_tensor = torch.as_tensor(self.high, device=self.device)
        self.dtype = self._torch_dtype
        self.device = self.low_tensor.device

    def to(self, device: Union[str, torch.device]):
        return type(self)(low=self.low, high=self.high, shape=self.shape, dtype=self.dtype, device=device)

    def to_(self, device: Union[str, torch.device]) -> None:
        self.low_tensor = self.low_tensor.to(device=device)
        self.high_tensor = self.high_tensor.to(device=device)
        self.device = device

    def sample(self):
        self.dtype = self._numpy_dtype
        sample = super().sample()
        self.dtype = self._torch_dtype
        return torch.as_tensor(sample, dtype=self._torch_dtype, device=self.device)

    def contains(self, x: Union[list, np.ndarray, Tensor]) -> bool:
        if isinstance(x, list):
            x = np.array(x)  # Promote list to array for contains check
        if isinstance(x, Tensor):
            return (
                x.shape == self.shape
                and (x >= self.low_tensor).all()
                and (x <= self.high_tensor).all()
            )
        return x.shape == self.shape and np.all(x >= self.low) and np.all(x <= self.high)

    def __repr__(self):
        return (
            f"{type(self).__name__}({self.low.min()}, {self.high.max()}, "
            f"{self.shape}, {self.dtype}"
            + (f", device={self.device}" if self.device is not None else "")
            + ")"
        )

    @classmethod
    def from_box(cls, box: spaces.Box, device: torch.device = None):
        return cls(
            low=box.low.flat[0],
            high=box.high.flat[0],
            shape=box.shape,
            dtype=box.dtype,  # NOTE: Gets converted in TensorSpace constructor.
            device=device,
        )


@batch_space.register(TensorBox)
def _(space: TensorBox, n: int = 1) -> TensorBox:
    repeats = tuple([n] + [1] * space.low.ndim)
    low, high = np.tile(space.low, repeats), np.tile(space.high, repeats)
    return type(space)(low=low, high=high, dtype=space.dtype, device=space.device)


@flatten_space.register(TensorBox)
def _(space: TensorBox):
    return type(space)(space.low.flatten(), space.high.flatten(), dtype=space.dtype, device=space.device)


class TensorDiscrete(TensorSpace, spaces.Discrete):
    def __init__(self, n: int, seed: int=None, device: torch.device=None, **kwargs):
        super().__init__(n=n, seed=seed, device=device, **kwargs)

    def contains(self, v: Union[int, Tensor]) -> bool:
        if isinstance(v, Tensor):
            v = v.detach().cpu().numpy()
        return super().contains(v)

    def sample(self):
        self.dtype = self._numpy_dtype
        s = super().sample()
        self.dtype = self._torch_dtype
        return torch.as_tensor(s, dtype=self.dtype, device=self.device)

    def __repr__(self) -> str:
        result_str = f"{type(self).__name__}({self.n}"
        if self.start != 0:
            result_str += f", start={self.start}"
        if self.device.type != "cpu":
            result_str += f", device={self.device}"
        return result_str + ")"


class TensorMultiDiscrete(TensorSpace, spaces.MultiDiscrete):
    def __init__(
        self,
        nvec: Sequence[int],
        dtype=np.int64,
        seed: Optional[int] = None,
        device: torch.device = None,
    ):
        # NOTE: Nvec can't be a GPU tensor, because the MultiDiscrete.__init__ makes it into a numpy
        # array.
        super().__init__(nvec=nvec, dtype=dtype, seed=seed, device=device)

    def contains(self, v: Tensor) -> bool:
        try:
            return super().contains(v)
        except:
            v_numpy = v.detach().cpu().numpy()
            return super().contains(v_numpy)

    def sample(self) -> torch.LongTensor:
        self.dtype = self._numpy_dtype
        s = super().sample()
        self.dtype = self._torch_dtype
        return torch.as_tensor(s, dtype=self.dtype, device=self.device)

    def __repr__(self) -> str:
        result_str = f"{type(self).__name__}({self.nvec}"
        if self.device.type != "cpu":
            result_str += f", device={self.device}"
        return result_str + ")"


@batch_space.register(TensorDiscrete)
def _batch_discrete_space(space: TensorDiscrete, n: int = 1) -> TensorMultiDiscrete:
    return TensorMultiDiscrete(torch.full((n,), space.n, dtype=space.dtype))
