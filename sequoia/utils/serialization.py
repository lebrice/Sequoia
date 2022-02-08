from dataclasses import dataclass, fields
from inspect import isfunction
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple, Type, TypeVar, Union, get_type_hints

import torch
from simple_parsing.helpers import Serializable as SerializableBase
from simple_parsing.helpers.serialization import register_decoding_fn

from sequoia.utils.generic_functions import detach

from .generic_functions.detach import detach
from .generic_functions.move import move
from .logging_utils import get_logger
from .utils import dict_union

register_decoding_fn(torch.device, torch.device)

T = TypeVar("T")
logger = get_logger(__file__)


def cpu(x: Any) -> Any:
    return move(x, "cpu")


class Pickleable:
    """Helps make a class pickleable."""

    def __getstate__(self):
        """We implement this to just make sure to detach the tensors if any
        before pickling.
        """
        # We use `vars(self)` to get all the attributes, not just the fields.
        state_dict = vars(self)
        return cpu(detach(state_dict))

    def __setstate__(self, state: Dict):
        # logger.debug(f"__setstate__ was called")
        self.__dict__.update(state)


S = TypeVar("S", bound="Serializable")


@dataclass
class Serializable(SerializableBase, Pickleable, decode_into_subclasses=True):  # type: ignore
    # NOTE: This currently doesn't add much compared to `Serializable` from simple-parsing apart
    # from not dropping the keys.

    def save(self, path: Union[str, Path], **kwargs) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Save to temp file, so we don't corrupt the save file.
        save_path_tmp = path.with_name(path.stem + "_temp" + path.suffix)
        # write out to the temp file.
        super().save(save_path_tmp, **kwargs)
        # Rename the temp file to the right path, overwriting it if it exists.
        save_path_tmp.replace(path)

    def detach(self: S) -> S:
        return type(self)(
            **detach(
                {
                    field.name: getattr(self, field.name)
                    for field in fields(self)
                    if field.metadata.get("to_dict", True)
                }
            )
        )

    def to(self, device: Union[str, torch.device]):
        """Returns a new object with all the attributes 'moved' to `device`.

        NOTE: This doesn't implement anything related to the other args like
        memory format or dtype.
        TODO: Maybe add something to convert everything that is a Tensor or
        numpy array to a given dtype?
        """
        return type(self)(**{name: move(item, device) for name, item in self.items()})

    def items(self) -> Iterable[Tuple[str, Any]]:
        for field in fields(self):
            yield field.name, getattr(self, field.name)

    def cpu(self):
        return self.to("cpu")

    def cuda(self, device: Union[str, torch.device] = None):
        return self.to(device or "cuda")

    def merge(self, other: "Serializable") -> "Serializable":
        """Overwrite values in `self` present in 'other' with the values from
        `other`.
        Also merges child elements recursively.

        Returns a new object, i.e. this doesn't modify `self` in-place.
        """
        self_dict = self.to_dict()
        if isinstance(other, SerializableBase):
            other = other.to_dict()
        elif not isinstance(other, dict):
            raise RuntimeError(f"Can't merge self with {other}.")
        return type(self).from_dict(dict_union(self_dict, other))


class decode:
    @staticmethod
    def register(fn_or_type: Type = None):
        """Decorator to be used to register a decoding function for a given type.

        This can be used in two different ways. The type annotation can either be
        explicit, like so:
        ```python
        @decode.register(SomeType)
        def decode_some_type(v: str):
           return SomeType(v)  # return an instance of SomeType from a string.
        ```
        or implicitly determined through the return type annotation, like so:
        ```
        @decode.register
        def decode_some_type(v: str) -> SomeType:
           (...)
        ```

        In the end, this just calls `register_decoding_fn(SomeType, decode_some_type)`.
        """

        def _wrapper(fn):
            if fn_or_type is not None:
                type_ = fn_or_type
            else:
                type_hints = get_type_hints(fn)
                if "return" not in type_hints:
                    raise RuntimeError(
                        f"Need to either explicitly pass a type to `register`, or use "
                        f"a return type annotation (e.g. `-> Foo:`) on the function!"
                    )
                type_ = type_hints["return"]
            register_decoding_fn(type_, fn)
            return fn

        if isfunction(fn_or_type):
            fn = fn_or_type
            fn_or_type = None
            return _wrapper(fn)
        return _wrapper
