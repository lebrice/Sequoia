""" Subclass of `spaces.Dict` that allows custom dtypes and uses type annotations.
"""
import inspect
import dataclasses
from inspect import isclass
from collections import OrderedDict
from collections.abc import Mapping as MappingABC
from dataclasses import (
    _PARAMS,
    Field,
    _DataclassParams,
    dataclass,
    fields,
    is_dataclass,
    make_dataclass,
)
import typing
from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
    KeysView,
    List,
    Mapping,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_type_hints,
)
from dataclasses import _is_classvar
from copy import deepcopy
import numpy as np
import gym
from gym import Space, spaces
from gym.vector.utils import batch_space, concatenate

from .sparse import batch_space, concatenate

M = TypeVar("M", bound=Mapping[str, Any])
S = TypeVar("S")
Dataclass = TypeVar("Dataclass")

try:
    from typing import get_origin
except ImportError:
    # Python 3.7's typing module doesn't have this `get_origin` function, so get it from
    # `typing_inspect`.
    from typing_inspect import get_origin


class TypedDictSpace(spaces.Dict, Mapping[str, Space], Generic[M]):
    """ Subclass of `spaces.Dict` that allows custom dtypes and uses type annotations.

    ## Examples:

    - Using it just like a regular spaces.Dict:

    >>> from gym.spaces import Box
    >>> s = TypedDictSpace(x=Box(0, 1, (4,), dtype=np.float64))
    >>> s
    TypedDictSpace(x:Box([0. 0. 0. 0.], [1. 1. 1. 1.], (4,), float64))
    >>> _ = s.seed(123)
    >>> s.sample()
    {'x': array([0.66528138, 0.33239426, 0.30337907, 0.92981861])}

    - Using it like a TypedDict: (This equivalent to the above)

    >>> class VisionSpace(TypedDictSpace):
    ...     x: Box = Box(0, 1, (4,), dtype=np.float64)  
    >>> s = VisionSpace()
    >>> s
    VisionSpace(x:Box([0. 0. 0. 0.], [1. 1. 1. 1.], (4,), float64))
    >>> _ = s.seed(123)
    >>> s.sample()
    {'x': array([0.66528138, 0.33239426, 0.30337907, 0.92981861])}
    
    - You can also overwrite the values from the type annotations by passing them to the
      constructor:

    >>> s = VisionSpace(x=spaces.Box(0, 2, (3,), dtype=np.int64))
    >>> s
    VisionSpace(x:Box([0 0 0], [2 2 2], (3,), int64))
    >>> _ = s.seed(123)
    >>> s.sample()
    {'x': array([1, 0, 0])}

    ### Using custom dtypes
    
    Can use any type here, as long as it can receive the samples from each space as
    keyword arguments.

    One good example of this is to use a `dataclass` as the custom dtype.
    You are strongly encouraged to use a dtype that inherits from the `Mapping` class
    from `collections.abc`, so that samples form your space can be handled similarly to
    regular dictionaries.

    >>> from collections import OrderedDict
    >>> s = TypedDictSpace(x=spaces.Box(0, 1, (4,), dtype=int), dtype=OrderedDict)
    >>> s
    TypedDictSpace(x:Box([0 0 0 0], [1 1 1 1], (4,), int64), dtype=<class 'collections.OrderedDict'>)
    >>> _ = s.seed(123)
    >>> s.sample()
    OrderedDict([('x', array([1, 0, 0, 1]))])

    ### Required items:
    
    If an annotation on the class doesn't have a default value, then it is treated as a
    required argument:
    
    >>> class FooSpace(TypedDictSpace):
    ...     a: spaces.Box = spaces.Box(0, 1, (4,), int)
    ...     b: spaces.Discrete
    >>> s = FooSpace()  # doesn't work!
    Traceback (most recent call last):
      ...
    TypeError: Space of type <class 'sequoia.common.spaces.typed_dict.FooSpace'> requires a 'b' item!
    >>> s = FooSpace(b=spaces.Discrete(5))
    >>> s
    FooSpace(a:Box([0 0 0 0], [1 1 1 1], (4,), int64), b:Discrete(5))

    NOTE: spaces can also inherit from each other!

    >>> class ImageSegmentationSpace(VisionSpace):
    ...     bounding_box: Box
    ...
    >>> s = ImageSegmentationSpace(
    ...     x=spaces.Box(0, 1, (2, 2), dtype=float),
    ...     bounding_box=spaces.Box(0, 4, (4, 2), dtype=int),
    ... )
    >>> s
    ImageSegmentationSpace(x:Box([[0. 0.]
     [0. 0.]], [[1. 1.]
     [1. 1.]], (2, 2), float64), bounding_box:Box([[0 0]
     [0 0]
     [0 0]
     [0 0]], [[4 4]
     [4 4]
     [4 4]
     [4 4]], (4, 2), int64))
    """

    def __init__(
        self, spaces: Mapping[str, Space] = None, dtype: Type[M] = dict, **spaces_kwargs
    ):
        """Creates the TypedDict space.
        
        Can either pass a dict of spaces, or pass the spaces as keyword arguments.

        Parameters
        ----------
        spaces : Mapping[str, Space], optional
            Dictionary mapping from strings to spaces, by default None
        dtype : Type[M], optional
            Type of outputs to return. By default `dict`, but this can also use any
            other dtype which will accept the values from each space as a keyword
            argument. 
            
            NOTE: This `dtype` is usually set to some dataclass type in Sequoia, such as
            `Observation`, `Rewards`, etc. (subclasses of `Batch`).  

            By default, `dtype` is just `dict`, and `space.sample()` will return simple
            dictionaries.

        Raises
        ------
        RuntimeError
            If both `spaces` and **kwargs are used.
        TypeError
            If the class has a type annotation for a space, and the required space isn't
            passed as an argument (emulating a required argument, in a way).
        """

        if spaces and spaces_kwargs:
            raise RuntimeError(f"Can only use one of `spaces` or **kwargs, not both.")
        spaces_from_args = spaces or spaces_kwargs

        # have to use OrderedDict just in case python <= 3.6.x
        spaces_from_annotations: Dict[str, gym.Space] = OrderedDict()

        cls = type(self)
        class_typed_attributes: Dict[str, Type] = get_type_hints(cls)
        # NOTE: This is only needed when using `__future__ import annotations` in a
        # client file:
        # Get the `globals` of the caller when checking type annotations:
        # TODO: Might actually need to get the globals of where that class is defined!
        # caller_globals = inspect.stack()[1][0].f_globals
        # class_typed_attributes: Dict[str, Type] = get_type_hints(cls, globalns=caller_globals)

        if class_typed_attributes:
            for attribute, type_annotation in class_typed_attributes.items():
                if _is_classvar(type_annotation, typing=typing):
                    continue

                is_space = False
                if isclass(type_annotation) and issubclass(type_annotation, gym.Space):
                    is_space = True
                else:
                    origin = get_origin(type_annotation)
                    is_space = (
                        origin is not None
                        and isclass(origin)
                        and issubclass(origin, gym.Space)
                    )

                # NOTE: emulate a 'required argument' when there is a type
                # annotation, but no value.
                # TODO: How about a None value, is that ok?
                if is_space:
                    _missing = object()
                    value = getattr(cls, attribute, _missing)
                    if value is _missing and attribute not in spaces_from_args:
                        raise TypeError(
                            f"Space of type {type(self)} requires a '{attribute}' item!"
                        )
                    if isinstance(value, gym.Space):
                        # Shouldn't be able to have two annotations with the same name.
                        assert attribute not in spaces_from_annotations
                        # TODO: Should copy the space, so that modifying the class
                        # attribute doesn't affect the instances of that space.
                        spaces_from_annotations[attribute] = deepcopy(value)

        # Avoid the annoying sorting of keys that `spaces.Dict` does if we pass a
        # regular dict.
        spaces = OrderedDict()  # Need to use this for 3.6.x
        spaces.update(spaces_from_annotations)
        spaces.update(
            spaces_from_args
        )  # Arguments overwrite the spaces from the annotations.

        if not spaces:
            raise TypeError(
                f"Need to either have type annotations on the class, or pass some "
                f"arguments to the constructor!"
            )
        assert all(isinstance(s, gym.Space) for s in spaces.values()), spaces

        super().__init__(spaces=spaces)
        self.spaces = dict(self.spaces)  # Get rid of the OrderedDict.

        # Sequoia-specific check.
        if "x" in self.spaces:
            assert list(self.spaces.keys()).index("x") == 0, self.spaces

        self.dtype = dtype

        # Optional: But just to make sure this works:
        if dataclasses.is_dataclass(self.dtype):
            dtype_fields: List[str] = [f.name for f in dataclasses.fields(self.dtype)]
            # Check that the dtype can handle all the entries of `self.spaces`, so that
            # we won't get any issues when calling `self.dtype(**super().sample())`.
            for space_name, space in self.spaces.items():
                if space_name not in dtype_fields:
                    raise RuntimeError(
                        f"dtype {self.dtype} doesn't have a field for space "
                        f"'{space_name}' ({space})!"
                    )

    def keys(self) -> Sequence[str]:
        return self.spaces.keys()

    def items(self) -> Iterable[Tuple[str, Space]]:
        return self.spaces.items()

    def values(self) -> Sequence[Space]:
        return self.spaces.values()

    def sample(self) -> M:
        dict_sample: dict = super().sample()
        # Gets rid of OrderedDict.
        return self.dtype(**dict_sample)

    def __getattr__(self, attr: str) -> Space:
        if attr != "spaces":
            if attr in self.spaces:
                return self.spaces[attr]
        raise AttributeError(f"Space doesn't have attribute {attr}")

    def __getitem__(self, key: Union[str, int]) -> Space:
        if key not in self.spaces:
            if isinstance(key, int):
                # IDEA: Try to get the item at given index in the keys? a bit like a
                # tuple space?
                # return self[list(self.spaces.keys())[key]]
                pass
        return super().__getitem__(key)

    def __len__(self) -> int:
        return len(self.spaces)

    # def __setitem__(self, key, value):
    #     return super().__setitem__(key, value)

    def contains(self, x: Union[M, Mapping[str, Space]]) -> bool:
        if is_dataclass(x):
            if is_dataclass(self.dtype):
                if not isinstance(x, self.dtype):
                    return False
            # NOTE: We don't use dataclasses.asdict as it doesn't work with Tensor
            # items with grad attributes.
            x = {f.name: getattr(x, f.name) for f in fields(x)}

        # NOTE: Modifying this so that we allow samples with more values, as long as it
        # has all the required keys.
        if not isinstance(x, dict) or not all(k in x for k in self.spaces):
            return False
        for k, space in self.spaces.items():
            if k not in x:
                return False
            if not space.contains(x[k]):
                return False
        return True
        # return super().contains(x)

    def __repr__(self) -> str:
        return (
            f"{str(type(self).__name__)}("
            + ", ".join([f"{k}:{s}" for k, s in self.spaces.items()])
            + (f", dtype={self.dtype}" if self.dtype is not dict else "")
            + ")"
        )

    def __eq__(self, other):
        if isinstance(other, TypedDictSpace) and self.dtype != other.dtype:
            return False
        return super().__eq__(other)


from functools import singledispatch


def _is_singledispatch(module_function):
    return hasattr(module_function, "registry")


def register_variant(module, module_fn_name: str):
    """ Converts a function from the given module to a singledispatch callable,
    and registers the wrapped function as the callable to use for Sparse spaces.
    
    The module function must have the space as the first argument for this to
    work.
    """
    module_function = getattr(module, module_fn_name)

    # Convert the function to a singledispatch callable.
    if not _is_singledispatch(module_function):
        module_function = singledispatch(module_function)
        setattr(module, module_fn_name, module_function)
    # Register the function as the callable to use when the first arg is a
    # Sparse object.
    def wrapper(function):
        module_function.register(TypedDictSpace, function)
        return function

    return wrapper


import gym.vector.utils
from gym.vector.utils.shared_memory import (
    read_from_shared_memory as read_from_shared_memory_,
)


@batch_space.register(TypedDictSpace)
# @register_variant(gym.vector.utils, "batch_space")
def _batch_typed_dict_space(space: TypedDictSpace, n: int = 1) -> spaces.Dict:
    return type(space)(
        {key: batch_space(subspace, n=n) for (key, subspace) in space.spaces.items()},
        dtype=space.dtype,
    )


@concatenate.register(TypedDictSpace)
# @register_variant(gym.vector.utils, "concatenate")
def _concatenate_typed_dicts(
    space: TypedDictSpace,
    items: Union[list, tuple],
    out: Union[tuple, dict, np.ndarray],
) -> Dict:
    return space.dtype(
        **{
            key: concatenate(subspace, [item[key] for item in items], out=out[key])
            for (key, subspace) in space.spaces.items()
        }
    )


def _add_field_to_dataclass(
    dataclass_type: Type[Dataclass],
    new_name: str,
    new_fields: List[Union[str, Tuple[str, Type], Tuple[str, Type, Field]]],
) -> Type[Dataclass]:
    """ Dynamically creates a new dataclass which adds `new_fields` to `dataclass_type`.
    
    NOTE: This probably shouldn't be used, in favor of having 
    """
    assert is_dataclass(dataclass_type)
    old_fields = [(f.name, f.type, f) for f in fields(dataclass_type)]
    bases = (dataclass_type,)
    dataclass_params: _DataclassParams = getattr(dataclass_type, _PARAMS)
    new_type = make_dataclass(
        new_name,
        fields=old_fields + new_fields,
        bases=bases,
        init=dataclass_params.init,
        repr=dataclass_params.repr,
        eq=dataclass_params.eq,
        order=dataclass_params.order,
        unsafe_hash=dataclass_params.order,
        frozen=dataclass_params.frozen,
    )
    return new_type
