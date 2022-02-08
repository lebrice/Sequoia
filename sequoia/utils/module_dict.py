""" Typed wrapper around `nn.ModuleDict`, just that just adds a get method. """
from typing import Any, MutableMapping, TypeVar, Union

from torch import nn

M = TypeVar("M", bound=nn.Module)
T = TypeVar("T")


class ModuleDict(nn.ModuleDict, MutableMapping[str, M]):
    def get(self, key: str, default: Any = None) -> Union[M, Any]:
        """Returns the module at `self[key]` if present, else `default`.

        Args:
            key (str): a key.
            default (Union[M, nn.Module], optional): Default value to return.
                Defaults to None.

        Returns:
            Union[Optional[nn.Module], Optional[M]]: The nn.Module at that key.
        """
        return self[key] if key in self else default
