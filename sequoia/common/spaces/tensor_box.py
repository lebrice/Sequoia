""" TODO: Maybe create a typed version of 'add_tensor_support' of gym_wrappers.convert_tensors
"""
import gym
from typing import Union, Optional
from gym import spaces
from torch import Tensor
import numpy as np
import torch

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


from abc import ABC

def supports_tensors(space: gym.Space) -> bool:
    pass

class TensorSpace(gym.Space, ABC):
    def __init__(self, *args, device: torch.device = None, **kwargs):
        # super().__init__(*args, **kwargs)
        self.device: Optional[torch.device] = torch.device(device) if device else None
        # Depending on the value passed to `dtype`
        dtype = kwargs.get("dtype")
        if dtype is None:
            if isinstance(self, (spaces.Discrete, spaces.MultiDiscrete)):
                # NOTE: They dont actually give a 'dtype' argument for these. 
                self._numpy_dtype = np.int64
                self._torch_dtype = torch.int64
            else:
                raise NotImplementedError(f"Not passing dtype to space {self}?")
        elif any(dtype == key for key in numpy_to_torch_dtypes):
            self._numpy_dtype = dtype
            self._torch_dtype = [
                v for k, v in numpy_to_torch_dtypes.items() if k == dtype
            ][0]
        elif any(dtype == key for key in torch_to_numpy_dtypes):
            self._numpy_dtype = [
                v for k, v in torch_to_numpy_dtypes.items() if k == dtype
            ][0]
            self._torch_dtype = dtype
        elif str(dtype) == "float32":
            self._numpy_dtype = np.float32
            self._torch_dtype = torch.float32
        else:
            assert not any(dtype == k for k in numpy_to_torch_dtypes)
            assert not any(dtype == k for k in torch_to_numpy_dtypes)
            raise NotImplementedError(
                f"Unsupported dtype {dtype} (of type {type(dtype)})"
            )
        if "dtype" in kwargs:
            kwargs["dtype"] = self._numpy_dtype
        super().__init__(*args, **kwargs)
        
        self.dtype: torch.dtype = self._torch_dtype


class TensorBox(TensorSpace, spaces.Box):
    """ Box space that accepts both Tensor and ndarrays. """

    def __init__(
        self, low, high, shape=None, dtype=np.float32, device: torch.device = None
    ):
        super().__init__(low, high, shape=shape, dtype=dtype, device=device)
        self.low_tensor = torch.as_tensor(self.low, device=self.device)
        self.high_tensor = torch.as_tensor(self.high, device=self.device)
        self.dtype = self._torch_dtype

    def sample(self):
        sample = super().sample()
        return torch.as_tensor(sample, device=self.device)

    def contains(self, x: Union[list, np.ndarray, Tensor]) -> bool:
        if isinstance(x, list):
            x = np.array(x)  # Promote list to array for contains check
        if isinstance(x, Tensor):
            return (
                x.shape == self.shape
                and (x >= self.low_tensor).all()
                and (x <= self.high_tensor).all()
            )
        return (
            x.shape == self.shape and np.all(x >= self.low) and np.all(x <= self.high)
        )

    def __repr__(self):
        return (
            f"{type(self).__name__}({self.low.min()}, {self.high.max()}, "
            f"{self.shape}, {self.dtype}, device={self.device})"
        )


class TensorDiscrete(TensorSpace, spaces.Discrete):
    def contains(self, v: Tensor) -> bool:
        v_numpy = v.detach().cpu().numpy()
        return super().contains(v_numpy)

    def sample(self):
        return torch.as_tensor(super().sample(), dtype=self.dtype, device=self.device)


class TensorMultiDiscrete(TensorSpace, spaces.MultiDiscrete):
    def contains(self, v: Tensor) -> bool:
        try:
            return super().contains(v)
        except:
            v_numpy = v.detach().cpu().numpy()
            return super().contains(v_numpy)

    def sample(self):
        s = super().sample()
        return torch.as_tensor(s, dtype=self.dtype, device=self.device)


from gym.vector.utils.spaces import batch_space


@batch_space.register(TensorDiscrete)
def _batch_discrete_space(space: TensorDiscrete, n: int = 1) -> TensorMultiDiscrete:
    return TensorMultiDiscrete(torch.full((n,), space.n, dtype=space.dtype))
