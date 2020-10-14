import inspect
import json
from collections import OrderedDict
from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import Enum
from functools import singledispatch
from io import StringIO
from pathlib import Path
from pprint import pprint
from typing import (Any, Callable, Dict, Iterable, List, Optional, Sequence,
                    Tuple, Type, TypeVar, Union)

import numpy as np
import torch
from torch import Tensor, nn

from simple_parsing.helpers import Serializable as SerializableBase
from simple_parsing.helpers import SimpleJsonEncoder, encode
from simple_parsing.helpers.serialization import encode, register_decoding_fn

from .detach import detach
from .encode import encode
from .logging_utils import get_logger
from .move import move
from .utils import dict_union

T = TypeVar("T")
logger = get_logger(__file__)

def cpu(x: Any) -> Any:
    return move(x, "cpu")

class Pickleable():
    """ Helps make a class pickleable. """
    def __getstate__(self):
        """ We implement this to just make sure to detach the tensors if any
        before pickling.
        """
        # We use `vars(self)` to get all the attributes, not just the fields.
        state_dict = vars(self)
        return cpu(detach(state_dict))
    
    def __setstate__(self, state: Dict):
        # logger.debug(f"__setstate__ was called")
        self.__dict__.update(state)


S = TypeVar("S", bound="Serializable")
from typing import ClassVar

@dataclass
class Serializable(SerializableBase, Pickleable, decode_into_subclasses=True):  # type: ignore
    # NOTE: This currently doesn't add much compared to `Serializable` from simple-parsing apart
    # from not dropping the keys.
    
    def save(self, path: Union[str, Path], **kwargs) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Save to temp file, so we don't corrupt the save file.
        save_path_tmp = path.with_name(path.stem + "_temp" + path.suffix)
        # write out to the temp file.
        super().save(save_path_tmp, **kwargs)
        # Rename the temp file to the right path, overwriting it if it exists.
        save_path_tmp.replace(path)

    def detach(self: S) -> S:
        return type(self).from_dict(detach(self.to_dict()))

    def to(self, device: Union[str, torch.device]):
        """Returns a new object with all the attributes 'moved' to `device`.

        NOTE: This doesn't implement anything related to the other args like
        memory format or dtype.
        TODO: Maybe add something to convert everything that is a Tensor or 
        numpy array to a given dtype?
        """
        return type(self).from_dict({
            name: move(item, device)
            for name, item in self.items()
        })

    def items(self) -> Iterable[Tuple[str, Any]]:
        for field in fields(self):
            yield field.name, getattr(self, field.name)

    def cpu(self):
        return self.to("cpu")

    def cuda(self, device: Union[str, torch.device]=None):
        return self.to(device or "cuda")

    
    def merge(self, other: "Serializable") -> "Serializable":
        """ Overwrite values in `self` present in 'other' with the values from
        `other`.
        Also merges child elements recursively.
        
        Returns a new object, i.e. this doesn't modify `self` in-place.
        """
        self_dict = self.to_dict()
        if isinstance(other, SerializableBase):
            other = other.to_dict()
        elif not isinstance(other, dict):
            raise RuntimeError(f"Can't merge self with {other}.")
        return type(self).from_dict(dict_union(self_dict, other))
