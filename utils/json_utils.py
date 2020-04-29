import json
from collections import OrderedDict
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, TypeVar, Union

from torch import nn

T = TypeVar("T")


def is_json_serializable(value: str):
    try:
        return json.loads(json.dumps(value)) == value 
    except:
        return False


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


def take_out_unsuported_values(d: Dict, weird_things: Tuple[Any] = (type,)) -> dict:
    """ Takes out values from the dict that aren't supported by Wandb. """
    result: Dict = OrderedDict()
    for key, value in d.items():
        new_value = value
        if isinstance(value, dict):
            new_value = take_out_unsuported_values(value)
        elif isinstance(value, weird_things):
            print(f"Value at key '{key}' is weird, not keeping it.")
            new_value = None
        elif isinstance(value, list):
            new_value = list(filter(lambda v: None if isinstance(v, weird_things) else v, value))      
        new_value = value
    return result


def try_load(path: Path, default: T=None) -> Optional[T]:
    try:
        if path.suffix == ".json":
            with open(path) as f:
                return json.load(f)
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
