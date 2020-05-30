import json
import logging
from collections import OrderedDict
from dataclasses import asdict, dataclass, fields, is_dataclass
from functools import singledispatch
from io import StringIO
from pathlib import Path
from pprint import pprint
from typing import (Any, Callable, Dict, Iterable, List, Optional, Tuple, Type,
                    TypeVar, Union)

import numpy as np
import torch
from simple_parsing.helpers import JsonSerializable as JsonSerializableBase
from simple_parsing.helpers import SimpleEncoder, encode
from simple_parsing.helpers.serialization import register_decoding_fn, from_dict
from torch import Tensor, nn

T = TypeVar("T")
logger = logging.getLogger(__file__)

register_decoding_fn(Tensor, torch.as_tensor)
register_decoding_fn(np.ndarray, np.asarray)

@dataclass
class JsonSerializable(JsonSerializableBase, decode_into_subclasses=True):  # type: ignore
    
    def dumps(self, *, sort_keys=True, **dumps_kwargs) -> str:
        dumps_kwargs.setdefault("sort_keys", sort_keys)
        return super().dumps(**dumps_kwargs)

    def save_json(self, path: Path, **dump_kwargs) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Save to temp file, so we don't corrupt the save file.
        save_path_tmp = path.with_suffix(".tmp")
        # write out to the temp file.
        with open(save_path_tmp, "w") as f:
            self.dump(f, **dump_kwargs)
        # Rename the temp file to the right path, overwriting it if it exists.
        save_path_tmp.replace(path)
        # super().save_json(path, **dump_kwargs)

    def __getstate__(self):
        """ We implement this to just make sure to detach the tensors if any
        before pickling.
        """
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        for key, value in state.items():
            if isinstance(value, Tensor):
                if value.requires_grad:
                    value = value.detach()
                state[key] = value.cpu()
        # Remove the unpicklable entries.
        return state
    
    def to_dict(self) -> Dict:
        return self.__getstate__()
        # self.detach()
        # if hasattr(self, "drop_tensors"):
        #     self.drop_tensors()
        # return super().to_dict()

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
            if isinstance(value, JsonSerializable):
                value = value.detach()
            setattr(self, key, value)
        return self

    def cpu(self) -> None:
        for key, value in vars(self).items():
            if isinstance(value, (Tensor, JsonSerializable)):
                value = value.cpu()
            setattr(self, key, value)


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


def is_json_serializable(value: str):
    if isinstance(value, JsonSerializable):
        return True
    elif type(value) in encode.registry:
        return True
    try:
        return json.loads(json.dumps(value, cls=SimpleEncoder)) == value 
    except:
        return False


def take_out_unsuported_values(d: Dict, default_value: Any=None) -> Dict:
    result: Dict = OrderedDict()
    for k, v in d.items():
        # logger.debug(f"key {k} with value {v} is json-serializable: {is_json_serializable(v)}")
        if is_json_serializable(v):
            result[k] = v
        elif isinstance(v, dict):
            result[k] = take_out_unsuported_values(v, default_value)
        else:
            result[k] = default_value
    return result


def get_new_file(file: Path) -> Path:
    """Creates a new file, adding _{i} suffixes until the file doesn't exist.
    
    Args:
        file (Path): A path.
    
    Returns:
        Path: a path that is new. Might have a new _{i} suffix.
    """
    if not file.exists():
        return file
    else:
        i = 0
        file_i = file.with_name(file.stem + f"_{i}" + file.suffix)
        while file_i.exists():
            i += 1
            file_i = file.with_name(file.stem + f"_{i}" + file.suffix)
        file = file_i
    return file

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
