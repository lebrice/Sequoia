""" Small 'patch' for the NamedTuple type, just so we can use
isinstance(obj, NamedTuple) and issubclass(some_class, NamedTuple) work
correctly.
"""
from inspect import isclass
from typing import Any, Type, NamedTuple
from typing import NamedTuple, Type

def is_namedtuple(obj: Any) -> bool:
    """ Taken from https://stackoverflow.com/a/62692640/6388696 """
    return (
            isinstance(obj, tuple) and
            hasattr(obj, '_asdict') and
            hasattr(obj, '_fields')
    )


def is_namedtuple_type(obj: Type) -> bool:
    """ Taken from https://stackoverflow.com/a/62692640/6388696 """
    return obj is NamedTuple or (
        isclass(obj) and issubclass(obj, tuple) and
            hasattr(obj, '_asdict') and
            hasattr(obj, '_fields')
    )


def _instance_check_for_namedtuples(self: Type[Type[NamedTuple]], instance: Type[NamedTuple]):
    # print(self, instance)
    if self is NamedTuple:
        return is_namedtuple(instance)
    return super().__instancecheck__(instance)  # type: ignore


def _subclass_check_for_namedtuples(self: Type[Type[NamedTuple]], subclass: Type[NamedTuple]):
    # print(self, subclass)
    if self is NamedTuple:
        return is_namedtuple_type(subclass)
    return super().__subclasscheck__(subclass)  # type: ignore


type(NamedTuple).__instancecheck__ = _instance_check_for_namedtuples
type(NamedTuple).__subclasscheck__ = _subclass_check_for_namedtuples

