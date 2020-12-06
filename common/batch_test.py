""" TODO: Write tests that demonstrate / verify that the `Batch` class works
correctly.
"""


from dataclasses import dataclass
from typing import Dict, Type, Any, Tuple
import pytest
import doctest
import numpy as np

from . import batch
from .batch import Batch
from torch import Tensor
import torch
from torch import Tensor
from typing import Optional

@dataclass(frozen=True)
class ForwardPass(Batch):
    x: Tensor
    h_x: Optional[Tensor] = None
    y_pred: Optional[Tensor] = None


@pytest.mark.parametrize(
    "batch_type, items_dict",
    [
        (
            ForwardPass,
            dict(x = torch.arange(10),
                 h_x = torch.arange(10) + 1,
                 y_pred = torch.arange(10) + 2),
        ),                   
    ]
)
def test_batch_behaves_like_a_dict(batch_type, items_dict):
    obj = batch_type(**items_dict)
    
    # NOTE: dicts, along with their .keys() and .values() are ordered as of py37
    
    for i, (k, v) in enumerate(obj.items()):
        original_value = items_dict[k]
        
        assert k == list(items_dict.keys())[i]  # key order is the same.
        assert (v == original_value).all()
        if isinstance(original_value, Tensor):
            assert v is original_value  # Tensors shouldn't be cloned or copied

        assert (obj[k] == v).all()  # values are the same.
        assert (obj[k] == getattr(obj, k)).all() # getattr same as __getitem__
        assert (obj[i] == v).all() # can also be indexed with ints like a tuple.


@pytest.mark.parametrize(
    "batch_type, items_dict",
    [
        (
            ForwardPass,
            dict(x = torch.arange(10),
                 h_x = torch.arange(10) + 1,
                 y_pred = torch.arange(10) + 2),
        ),                   
    ]
)
def test_to(batch_type: Type[Batch], items_dict: Dict[str, Tensor]):
    """ Test that the 'to' method behaves like `torch.Tensor.to`, so that we
    can move all the items in a `Batch` between devices or dtypes.
    """
    original_devices: Dict[str, torch.device] = {
        k: v.device for k, v in items_dict.items()
    }
    original_dtypes: Dict[str, torch.dtype] = {
        k: v.dtype for k, v in items_dict.items()
    }

    obj = batch_type(**items_dict)

    # The devices and dtypes remain the same when creating the Batch with the
    # given items.
    for k, v in obj.items():
        original_value = items_dict[k]
        assert v.device == original_value.device == original_devices[k]
        assert v.dtype == original_value.dtype == original_dtypes[k]

    
    devices = tuple(original_devices.values())
    dtypes = tuple(original_dtypes.values())    
    # The 'devices' and 'dtypes' attributes give the devices and dtypes of all
    # items.
    assert obj.devices == devices
    assert obj.dtypes == dtypes

    if len(set(devices)) == 1:
        # If they all share the same device, then the `device` attribute on the
        # `batch` is this shared device.
        common_device = devices[0]
        assert obj.device == common_device

    if len(set(dtypes)) == 1:
        # If all tensors have the same dtype, then the `dtype` attribute on the
        # `batch` is this shared dtype.
        common_dtype = dtypes[0]
        assert obj.dtype == common_dtype

    # Test moving to another device, if possible.
    if torch.cuda.is_available():
        cuda_obj = obj.to("cuda")        
        for i, (k, v) in enumerate(cuda_obj.items()):
            assert v.device.type == "cuda"

    float_obj = obj.to(dtype=torch.float32)
    for k, v in float_obj.items():
        original_value = items_dict[k]
        assert v.device == original_value.device
        assert v.dtype == torch.float32
        assert (v == original_value.to(dtype=torch.float32)).all()


@pytest.mark.parametrize(
    "batch_type, items_dict",
    [
        (
            ForwardPass,
            dict(x = torch.arange(25).reshape([5, 5]),
                 h_x = torch.arange(25).reshape([5, 5]) + 1,
                 y_pred = torch.arange(25).reshape([5, 5]) + 2),
        ),
    ]
)
@pytest.mark.parametrize("index", [
    (0, 0), # obj[0, 0]
    (0, ..., 0), # obj[0, ..., 0]
    (slice(None), 0), # obj[:, 0]
    (slice(None), slice(3)), # obj[:, :3]
    (slice(None), slice(None, -3)), # obj[:, -3:]
    (slice(None), slice(None, None, 2)), # obj[:, ::2]
    (slice(None), np.arange(5) % 2 == 0), # obj[:, even_mask]
    (slice(None), np.arange(5) % 2 == 0), # obj[:, even_mask]
])
def test_tuple_indexing(batch_type: Type[Batch], items_dict: Dict[str, Tensor], index: Tuple[Any, ...]):
    """ Test that we can index into the object in the same style as an ndarray
    """
    obj = batch_type(**items_dict)
    
    
    keys = list(items_dict.keys())
    print(f"Expected keys: {keys}")
    expected_items = {
        k: items_dict[k][index[1:]] for k in np.array(keys)[index[0]] 
    }
    
    print(f"expected sliced items:")
    for key, value in expected_items.items():
        print(key, value)
    
    actual_slice = obj[index]
    
    if index[0] == slice(None):
        # actual_slice: Batch
        assert isinstance(actual_slice, batch_type)
        assert list(actual_slice.keys()) == keys

        for k, sliced_value in actual_slice.items():
            print(f"key {k}, index {index}")
            print(f"Sliced value: {sliced_value}")
            expected_value = expected_items[k]
            print(f"Expected value: {expected_value}")
            assert (sliced_value == expected_value).all()
    
    if isinstance(index[0], int):
        # e.g. Observations[0, <...>]
        key = keys[index[0]]
        expected_value = expected_items[key]
        assert (actual_slice == expected_value).all()
