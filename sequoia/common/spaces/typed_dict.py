""" IDEA: Dict space that supports .getattr """
import dataclasses
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
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Mapping,
    Iterable,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    KeysView,
)

import numpy as np
from gym import Space, spaces
from gym.vector.utils import batch_space, concatenate
from collections import OrderedDict

M = TypeVar("M", bound=Mapping[str, Any])
S = TypeVar("S")
Dataclass = TypeVar("Dataclass")


class TypedDictSpace(spaces.Dict, Mapping[str, Space], Generic[M]):
    def __init__(
        self, spaces: Mapping[str, Space] = None, dtype: Type[M] = dict, **spaces_kwargs
    ):
        # Avoid the annoying sorting of keys that `spaces.Dict` does if we pass a
        # regular dict.
        spaces = spaces or spaces_kwargs
        if spaces is not None and not isinstance(spaces, OrderedDict):
            spaces = OrderedDict(list(spaces.items()))
        super().__init__(spaces=spaces)
        self.spaces = dict(self.spaces)  # Get rid of the OrderedDict.

        if "x" in self.spaces:
            assert list(self.spaces.keys()).index("x") == 0, self.spaces

        if not (issubclass(dtype, MappingABC) or dataclasses.is_dataclass(dtype)):
            raise RuntimeError(
                f"`dtype` needs to be either a type of Mapping or a dataclass, got "
                f"{dtype})."
            )
        self.dtype = dtype
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
        if self.dtype is dict:
            return dict(dict_sample)  # Get rid of OrderedDict.
        return self.dtype(**dict_sample)

    def __getattr__(self, attr: str) -> Space:
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
        return super().contains(x)

    def __repr__(self) -> str:
        return (
            f"{str(type(self).__name__)}("
            + ", ".join([f"{k}:{s}" for k, s in self.spaces.items()])
            + f", dtype={self.dtype}"
            + ")"
        )

    def __eq__(self, other):
        if isinstance(other, TypedDictSpace) and self.dtype != other.dtype:
            return False
        return super().__eq__(other)


@batch_space.register(TypedDictSpace)
def _batch_typed_dict_space(space: TypedDictSpace, n: int = 1) -> spaces.Dict:
    return type(space)(
        {key: batch_space(subspace, n=n) for (key, subspace) in space.spaces.items()},
        dtype=space.dtype,
    )


@concatenate.register(TypedDictSpace)
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
