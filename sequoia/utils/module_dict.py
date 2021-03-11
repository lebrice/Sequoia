""" Typed wrapper around `nn.ModuleDict`, just that just adds a get method. """
from typing import MutableMapping, Optional, TypeVar, Union, overload
from torch import nn

M = TypeVar("M", bound=nn.Module)


class ModuleDict(nn.ModuleDict, MutableMapping[str, M]):
    @overload
    def get(self, key: str, default: nn.Module) -> nn.Module:
        ...

    @overload
    def get(self, key: str, default: M) -> M:
        ...

    @overload
    def get(self, key: str) -> nn.Module:
        ...

    def get(
        self, key: str, default: Union[M, nn.Module] = None
    ) -> Union[Optional[nn.Module], Optional[M]]:
        """Returns the module at `self[key]` if present, else `default`.

        Args:
            key (str): a key.
            default (Union[M, nn.Module], optional): Default value to return.
                Defaults to None.

        Returns:
            Union[Optional[nn.Module], Optional[M]]: The nn.Module at that key.
        """
        return self[key] if key in self else default
