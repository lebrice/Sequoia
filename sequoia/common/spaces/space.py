""" Small typing improvements to the `gym.spaces.Space` class. """
from gym.spaces import Space as _Space
from typing import TypeVar, Generic, Union, Any

T = TypeVar("T")

class Space(_Space, Generic[T]):
    def sample(self) -> T:
        return super().sample()

    def __contains__(self, x: Union[T, Any]) -> bool:
        return super().__contains__(x)

    def contains(self, v: Union[T, Any]) -> bool:
        return super().contains(v)
