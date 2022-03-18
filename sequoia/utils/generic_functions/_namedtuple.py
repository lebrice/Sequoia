""" Small 'patch' for the NamedTuple type, just so we can use
isinstance(obj, NamedTuple) and issubclass(some_class, NamedTuple) work
correctly.
"""
from inspect import isclass
from typing import Any, NamedTuple, Type


def is_namedtuple(obj: Any) -> bool:
    """Taken from https://stackoverflow.com/a/62692640/6388696"""
    return isinstance(obj, tuple) and hasattr(obj, "_asdict") and hasattr(obj, "_fields")


def is_namedtuple_type(obj: Type) -> bool:
    """Taken from https://stackoverflow.com/a/62692640/6388696"""
    return obj is NamedTuple or (
        isclass(obj)
        and issubclass(obj, tuple)
        and hasattr(obj, "_asdict")
        and hasattr(obj, "_fields")
    )
