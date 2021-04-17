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


class TensorBox(spaces.Box):
    """ Box space that accepts both Tensor and ndarrays. """

    def __init__(
        self, low, high, shape=None, dtype=np.float32, device: torch.device = None
    ):
        if dtype in numpy_to_torch_dtypes:
            self._numpy_dtype = dtype
            self._torch_dtype = numpy_to_torch_dtypes[dtype]
        elif dtype in torch_to_numpy_dtypes:
            self._numpy_dtype = torch_to_numpy_dtypes[dtype]
            self._torch_dtype = dtype
        elif str(dtype) == "float32":
            self._numpy_dtype = np.float32
            self._torch_dtype = torch.float32
        else:
            raise NotImplementedError(
                f"Unsupported dtype {dtype} (of type {type(dtype)})"
            )
        super().__init__(low, high, shape=shape, dtype=self._numpy_dtype)
        self.device: Optional[torch.device] = torch.device(device) if device else None

    def sample(self):
        sample = super().sample()
        return torch.as_tensor(sample, dtype=self._torch_dtype, device=self.device)

    def contains(self, x: Union[list, np.ndarray, Tensor]) -> bool:
        if isinstance(x, list):
            x = np.array(x)  # Promote list to array for contains check
        return (
            x.shape == self.shape and np.all(x >= self.low) and np.all(x <= self.high)
        )

    def __repr__(self):
        return f"{type(self).__name__}({self.low.min()}, {self.high.max()}, {self.shape}, {self.dtype}, device={self.device})"