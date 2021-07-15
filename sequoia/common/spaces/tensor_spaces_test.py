from gym import spaces
import numpy as np
import torch
from torch import Tensor
import pytest
from .tensor_spaces import TensorBox, numpy_to_torch_dtypes


@pytest.mark.parametrize("np_dtype", [np.uint8, np.float32])
def test_tensor_box(np_dtype: np.dtype):
    torch_dtype = numpy_to_torch_dtypes[np_dtype]

    space = spaces.Box(0, 1, (28, 28), dtype=np_dtype)
    new_space = TensorBox.from_box(space)
    sample = new_space.sample()

    assert isinstance(sample, Tensor)
    assert sample in new_space
    assert sample.cpu().numpy().astype(np_dtype) in space
    assert sample.dtype == torch_dtype
