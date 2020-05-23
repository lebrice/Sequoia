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
from simple_parsing.helpers.serialization import register_decoding_fn
from torch import Tensor, nn

T = TypeVar("T")
logger = logging.getLogger(__file__)

register_decoding_fn(Tensor, torch.as_tensor)
register_decoding_fn(np.ndarray, np.asarray)

@dataclass
class JsonSerializable(JsonSerializableBase, decode_into_subclasses=True):
    pass


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
