""" Generic function for replacing items in an object. """

import dataclasses
from typing import Dict, Any, TypeVar, Tuple, List
from collections.abc import Sequence, Mapping, MutableMapping
from functools import singledispatch
from . import NamedTuple
from gym import Space, spaces
T = TypeVar("T")


class Dataclass(type):
    """ Used so we can do `isinstance(obj, Dataclass)`, or maybe even
    register dataclass handlers for singledispatch generic functions.
    """
    def __instancecheck__(self, instance):
        # Return true if instance should be considered a (direct or indirect)
        # instance of class. If defined, called to implement
        # isinstance(instance, class).
        return dataclasses.is_dataclass(instance)

    def __subclasscheck__(self, subclass):
        # Return true if subclass should be considered a (direct or indirect)
        # subclass of class. If defined, called to implement
        # issubclass(subclass, class).
        return dataclasses.is_dataclass(subclass)


@singledispatch
def replace(obj: T, **items) -> T:
    """ Replaces the value at `key` in `obj` with `new_value`. Returns the
    modified object, either in-place (same instance as obj) or new.
    """
    raise NotImplementedError(
        f"TODO: Don't know how to set items '{items}' in obj {obj}, "
        f"(no handler registered for objects of type {obj}."
    )


@replace.register(Dataclass)
def _replace_dataclass_attribute(obj: Dataclass, **items) -> Dataclass:
    assert dataclasses.is_dataclass(obj)
    return dataclasses.replace(obj, **items)


@replace.register(NamedTuple)
def _replace_namedtuple_item(obj: Dataclass, **items) -> Dataclass:
    assert isinstance(obj, NamedTuple)
    return obj._replace(**items)


@replace.register(dict)
def _replace_dict_item(obj: Dict, **items) -> Dict:
    assert isinstance(obj, dict)
    assert all(key in obj for key in items), (
        "replace should only be used to replace items, not to add new ones."
    )
    new_obj = obj.copy()
    new_obj.update(items)
    return new_obj


@replace.register(list)
@replace.register(tuple)
def _replace_sequence_items(obj: Sequence, **items) -> Tuple:
    return type(obj)(
        items[i] if i in items else val
        for i, val in enumerate(obj)
    )


@replace.register
def _replace_dict_items(obj: spaces.Dict, **items) -> Dict:
    """ Handler for Dict spaces. """
    return type(obj)(replace(obj.spaces, **items))

