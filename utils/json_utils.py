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
from enum import Enum
import numpy as np
import torch
from simple_parsing.helpers import Serializable as SerializableBase
from simple_parsing.helpers import encode, SimpleJsonEncoder
from simple_parsing.helpers.serialization import register_decoding_fn
from torch import Tensor, nn

T = TypeVar("T")
logger = logging.getLogger(__file__)

register_decoding_fn(Tensor, torch.as_tensor)
register_decoding_fn(np.ndarray, np.asarray)

from simple_parsing.helpers.serialization import encode

@dataclass
class ModelStateDict(Dict[str, Tensor]):
    def __init__(self, path: Optional[Union[Path]]):
        if isinstance(path, Path):
            state_dict: Dict[str, Tensor] = torch.load(str(path))
        super().__init__(state_dict)
    
    def save(self, path: Path):
        # TODO: A bit of a stretch, but we could detect when a field that is
        # supposed to be, say a state dict is instead a Path, and just load it
        # from there!
        model_state_dict: Dict[str, Tensor] = OrderedDict()
        if save_model_weights:
            for k, tensor in self.model.state_dict().items():
                model_state_dict[k] = tensor.detach().cpu()

@encode.register
def encode_tensor(obj: Tensor) -> List:
    return obj.detach().cpu().tolist()


@encode.register
def encode_ndarray(obj: np.ndarray) -> List:
    return obj.tolist()



@dataclass
class Serializable(SerializableBase, decode_into_subclasses=True):  # type: ignore
    
    def save(self, path: Union[str, Path], **kwargs) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Save to temp file, so we don't corrupt the save file.
        save_path_tmp = path.with_name(path.stem + "_temp" + path.suffix)
        # write out to the temp file.
        super().save(save_path_tmp, **kwargs)
        # Rename the temp file to the right path, overwriting it if it exists.
        save_path_tmp.replace(path)

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


def is_json_serializable(value: str):
    if isinstance(value, Serializable):
        return True
    elif type(value) in encode.registry:
        return True
    try:
        return json.loads(json.dumps(value, cls=SimpleJsonEncoder)) == value 
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
