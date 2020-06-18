import json
from collections import OrderedDict
from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import Enum
from functools import singledispatch
from io import StringIO
from pathlib import Path
from pprint import pprint
from typing import (Any, Callable, Dict, Iterable, List, Optional, Tuple, Type,
                    TypeVar, Union)

import numpy as np
import torch
from torch import Tensor, nn

from simple_parsing.helpers import Serializable as SerializableBase
from simple_parsing.helpers import SimpleJsonEncoder, encode
from simple_parsing.helpers.serialization import encode, register_decoding_fn
from utils.logging_utils import get_logger

T = TypeVar("T")
logger = get_logger(__file__)

register_decoding_fn(Tensor, torch.as_tensor)
register_decoding_fn(np.ndarray, np.asarray)


@dataclass
class ModelStateDict(Dict[str, Tensor]):
    """ TODO: @lebrice Unused for now, was thinking of using this to make it easier to
    save/load model dict, but it's already pretty simple.

    # TODO: A bit of a stretch, but we could detect when a field that is
    # supposed to be, say a state dict is instead a Path, and just load it
    # from there!
    """ 
    def __init__(self, path: Optional[Union[Path]]):
        if isinstance(path, Path):
            state_dict: Dict[str, Tensor] = torch.load(str(path))
        super().__init__(state_dict)
    
    def save(self, path: Path):
        model_state_dict: Dict[str, Tensor] = {
            k: v.detach().cpu() for k, v in self.items()
        }


class Pickleable():
    """ Helps make a class pickleable. """
    def __getstate__(self):
        """ We implement this to just make sure to detach the tensors if any
        before pickling.
        """
        logger.debug(f"__getstate__ was called.")
        # Use `vars(self)`` to get all the attributes, not just the fields.
        d = vars(self)
        # Overwrite with `self.to_dict()` so we get fields in nice format.
        d.update(self.to_dict())
        return d

    def __setstate__(self, state: Dict):
        logger.debug(f"setstate was called")
        raise NotImplementedError("TODO: not implemented yet...")
        pass

    def detach(self):
        """Move all tensor attributes to the CPU and then detach them in-place.
        Returns `self`, for convenience.
        NOTE: also recursively moves and detaches `JsonSerializable` attributes.
        """
        logger.debug(f"Detaching the object of type {type(self)}")
        self.cpu()
        self.detach_()
        return self

    def detach_(self):
        """ Detaches all the Tensor attributes in-place, then returns `self`.
        
        NOTE: also recursively detaches `JsonSerializable` attributes.
        """
        for key, value in vars(self).items():
            if isinstance(value, Tensor):
                value = value.detach()
            if isinstance(value, Serializable):
                value = value.detach()
            setattr(self, key, value)
        return self

    def cpu(self) -> None:
        for key, value in vars(self).items():
            if isinstance(value, (Tensor, Serializable)):
                value = value.cpu()
            setattr(self, key, value)


@dataclass
class Serializable(SerializableBase, Pickleable, decode_into_subclasses=True):  # type: ignore
    # NOTE: This currently doesn't add much compared to `Serializable` from simple-parsing.
    
    def save(self, path: Union[str, Path], **kwargs) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Save to temp file, so we don't corrupt the save file.
        save_path_tmp = path.with_name(path.stem + "_temp" + path.suffix)
        # write out to the temp file.
        super().save(save_path_tmp, **kwargs)
        # Rename the temp file to the right path, overwriting it if it exists.
        save_path_tmp.replace(path)

    

@encode.register
def encode_tensor(obj: Tensor) -> List:
    return obj.detach().cpu().tolist()


@encode.register
def encode_ndarray(obj: np.ndarray) -> List:
    return obj.tolist()


@encode.register
def encode_tensor(v: Tensor) -> List:
    return v.tolist()


@encode.register
def encode_array(v: np.ndarray) -> List:
    return v.tolist()


@encode.register
def encode_path(obj: Path) -> str:
    return str(obj)


@encode.register
def encode_device(obj: torch.device) -> str:
    return str(obj)


@encode.register
def encode_enum(value: Enum):
    return value.value


def try_load(path: Path, default: T=None) -> Optional[T]:
    try:
        if path.suffix == ".json":
            with open(path) as f:
                return json.loads(f.read())
        elif path.suffix == ".csv":
            import numpy as np
            with open(path) as f:
                return np.loadtxt(f)
        elif path.suffix == ".pt":
            import torch
            with open(path, 'rb') as fb:
                return torch.load(fb)
        elif path.suffix == ".yml":
            import yaml
            with open(path) as f:
                return yaml.load(f)
        elif path.suffix == ".pkl":
            import pickle
            with open(path, 'rb') as fb:
                return pickle.load(fb)
        else:
            raise RuntimeError(f"Unable to load path {path}, unsupported extension.")
    except Exception as e:
        print(f"couldn't load path {path}: {e}")
        return default

    
# a = {
#     "bob.txt": Path("bob.txt")
# }
# s = dumps(a)
# print(s)
# b = loads(s)
# print(b)
# exit()
