""" Registers more datatypes to be used by the 'encode' function from
simple-parsing when serializing objects to json or yaml.
"""
import enum
import inspect
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from torch import Tensor

from simple_parsing.helpers import encode
from simple_parsing.helpers.serialization import register_decoding_fn

# Register functions for decoding Tensor and ndarray fields from json/yaml.
register_decoding_fn(Tensor, torch.as_tensor)
register_decoding_fn(np.ndarray, np.asarray)


@encode.register(Tensor)
@encode.register(np.ndarray)
def encode_tensor(obj: Union[Tensor, np.ndarray]) -> List:
    return obj.tolist()


@encode.register
def encode_type(obj: type) -> List:
    if inspect.isclass(obj):
        return str(obj.__qualname__)
    elif inspect.isfunction(obj):
        return str(obj.__name__)
    return str(obj)


@encode.register
def encode_path(obj: Path) -> str:
    return str(obj)


@encode.register
def encode_device(obj: torch.device) -> str:
    return str(obj)


@encode.register
def encode_enum(value: enum.Enum):
    return value.value
