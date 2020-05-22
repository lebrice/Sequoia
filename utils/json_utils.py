import json
from collections import OrderedDict
from dataclasses import asdict, is_dataclass, fields
from functools import singledispatch
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypeVar, Union, Callable
from simple_parsing.helpers import from_dict, JsonSerializable as JsonSerializableBase
from torch import nn, Tensor
import numpy as np
import logging
logger = logging.getLogger(__file__)
from io import StringIO
            
T = TypeVar("T")


class MyEncoder(json.JSONEncoder):
    def default(self, o: Any):
        return encode(o)

@singledispatch
def encode(obj: Any) -> Dict:
    try:
        if is_dataclass(obj):
            d: Dict = OrderedDict()
            for field in fields(obj):
                value = getattr(obj, field.name)
                d[field.name] = encode(value) 
            return d
        elif hasattr(obj, "__dict__"):
            return obj.__dict__
        else:
            return obj
    except Exception as e:
        logger.debug(f"Cannot encode object {obj}: {e}")
        raise e

@encode.register
def encode_tensor(v: Tensor) -> List:
    return v.tolist()

@encode.register
def encode_array(v: np.ndarray) -> List:
    return v.tolist()

@encode.register
def encode_path(obj: Path) -> str:
    return str(obj)

object_hooks: List[Callable[[Any], Any]] = []

object_hooks.append(Path)

def dumps(v: Any) -> str:
    return json.dumps(v, cls=MyEncoder)

def loads(s: str) -> Optional[Any]:
    for obj_hook in object_hooks:
        try:
            return json.loads(s, object_hook=obj_hook)
        except:
            pass
    return json.loads(s)


def is_json_serializable(value: str):
    if isinstance(value, JsonSerializable):
        return True
    elif type(value) in encode.registry:
        return True
    try:
        return loads(json.dumps(value, cls=MyEncoder)) == value 
    except:
        return False

def take_out_unsuported_values(d: Dict, default_value: Any=None) -> Dict:
    result: Dict = OrderedDict()
    for k, v in d.items():
        if is_json_serializable(v):
            result[k] = v
        elif isinstance(v, dict):
            result[k] = take_out_unsuported_values(v, default_value)
        else:
            result[k] = default_value
    return result

from dataclasses import asdict
from pprint import pprint
from utils.json_utils import MyEncoder, take_out_unsuported_values

class JsonSerializable(JsonSerializableBase):
    def save_json(self, path: Union[Path, str], indent: Union[int, str]=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self, f, indent=indent, cls=MyEncoder)

    @classmethod
    def load_json(cls, path: Union[Path, str]):
        with open(path) as f:
            args_dict = loads(f.read())
        return from_dict(cls, args_dict)
    
    @classmethod
    def try_load_json(cls, path: Union[Path, str]):
        try:
            return cls.load_json(path)
        except Exception as e:
            print(f"Unable to load json ({e}), returning None.")
            return None


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


def to_str_dict(d: Dict) -> Dict[str, Union[str, Dict]]:
    for key, value in list(d.items()):
        d[key] = to_str(value) 
    return d


def to_str(value: Any) -> Any:
    try:
        return json.dumps(value)
    except Exception as e:
        if is_dataclass(value):
            value = asdict(value)
            return to_str_dict(value)
        elif isinstance(value, dict):
            return to_str_dict(value)
        elif isinstance(value, Path):
            return str(value)
        elif isinstance(value, nn.Module):
            return None
        elif isinstance(value, Iterable):
            return list(map(to_str, value))
        else:
            print("Couldn't make the value into a str:", value, e)
            return repr(value)


def try_load(path: Path, default: T=None) -> Optional[T]:
    try:
        if path.suffix == ".json":
            with open(path) as f:
                return loads(f.read())
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
