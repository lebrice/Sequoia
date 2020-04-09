import json
from collections import OrderedDict
from typing import Dict, Union, Any, Iterable, Tuple
from dataclasses import asdict, is_dataclass
from pathlib import Path
from torch import nn


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
