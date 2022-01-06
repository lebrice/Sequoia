"""Higher-order spaces, where the samples are spaces.

This can be useful to describe the kind of environments where a given algorithm can be applied,
for instance.

```python
# Algorithm can be used with simple environments with small observation spaces and discrete
# actions:
algo_space: Space[Env] = EnvsWhere(
    observation_space=Boxes(shape=lambda shape: np.prod(shape) < 100),
    action_space=Discretes(n=lambda n: n < 10),
)

cartpole = gym.make("CartPole-v1")
assert cartpole in algo_space
```
"""
from __future__ import annotations

import inspect
from abc import abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    Sequence,
)

import gym
from gym.envs.registration import EnvRegistry
import numpy as np
from gym import Space, spaces
from typing import cast


T = TypeVar("T")
T_cov = TypeVar("T_cov", covariant=True)
T_cot = TypeVar("T_cot", contravariant=True)
Predicate = Callable[[T_cot], bool]
Shape = Tuple[int, ...]
DType = Union[np.dtype, Type]
Bound = Union[float, np.ndarray]
S = TypeVar("S", bound=Space)
B = TypeVar("B", bound=spaces.Box)


class HigherSpace(Space, Generic[S]):
    @abstractmethod
    def contains(self, x: Any) -> bool:
        """Returns wether `x` is a Space that is described by this space."""

    @abstractmethod
    def sample(self) -> S:
        """Samples a space from this space of spaces."""

    def where(self, *assumptions: Predicate[S]) -> SpaceWithAssumptions[S]:
        """Adds assumptions to this space, restricting the kind of samples which are considered
        within it.

        ```python
        space: Space[Discrete] = Discretes().where(lambda n: n % 2 == 0)

        assert Discrete(2) in space
        assert Discrete(3) not in space
        ```
        """
        return SpaceWithAssumptions(self, assumptions=list(assumptions))


class Boxes(HigherSpace[B]):
    """Space of Box spaces."""

    def __init__(
        self,
        low: Union[Bound, Predicate[Bound]] = None,
        high: Union[Bound, Predicate[Bound]] = None,
        shape: Union[Shape, Predicate[Sequence[int]]] = None,
        dtype: Union[DType, Predicate[DType]] = None,
        space_dtype: Type[B] = spaces.Box,
    ) -> None:
        super().__init__(shape=None, dtype=None)
        self._shape = shape
        self._dtype = dtype
        self.low = low
        self.high = high
        self.space_dtype = space_dtype

    @property
    def shape(self) -> Optional[Union[Shape, Predicate[Shape]]]:
        return self._shape

    @property
    def dtype(self) -> Optional[Union[DType, Predicate[DType]]]:
        return self._dtype

    @dtype.setter
    def dtype(self, v: Optional[Union[DType, Predicate[DType]]]) -> None:
        self._dtype = v

    def __repr__(self) -> str:
        return f"{type(self).__name__}(low={self.low}, high={self.high}, shape={self.shape}, dtype={self.dtype})"

    def sample(self) -> B:
        """Sample a box space from this distribution."""
        if isinstance(self.low, Space) and isinstance(self.high, Space):
            # TODO: Doesn't quite work: Would need to make sure that low <= high everywhere.
            return self.space_dtype(low=self.low.sample(), high=self.high.sample())
        raise NotImplementedError()

    def contains(self, x: Any) -> bool:
        if not isinstance(x, spaces.Box):
            return False
        return all(
            [
                self._matches_dtype(x),
                self._matches_shape(x),
                self._matches_bounds(x),
            ]
        )

    def _matches_dtype(self, space: spaces.Box) -> bool:
        if self.dtype is None:
            return True
        if inspect.isclass(self.dtype) and not issubclass(space.dtype, self.dtype):
            return False
        if callable(self.dtype):
            dtype: Predicate[DType] = self.dtype
            return dtype(space.dtype)
        # Check the compatibility of np/torch dtypes, e.g. float32 <= float64
        return np.can_cast(space.dtype, self.dtype)

    def _matches_shape(self, space: spaces.Box) -> bool:
        if self.shape is None:
            return True

        if isinstance(self.shape, tuple):
            # Check if the shapes can be broadcasted / match somehow?
            try:
                np.broadcast_to(space.low, self.shape)
            except ValueError:
                return False
            else:
                return True
        # NOTE: Should we call the predicate if the space doesn't have a shape?
        if callable(self.shape) and space.shape is not None:
            shape: Predicate[Shape] = self.shape  # linting bug.
            if not shape(space.shape):
                return False
        return True

    def _matches_bounds(self, space: spaces.Box) -> bool:
        if not self._matches_shape(space):
            return False
        if self.low is None and self.high is None:
            return True
        # The lower bound of the space must be higher than our lower bound:
        o_low: np.ndarray = space.low
        if callable(self.low):
            if not self.low(o_low):
                return False
        elif self.low is not None:
            s_low = cast(Union[float, np.ndarray], self.low)
            if not np.greater_equal(s_low, o_low).all():
                return False
        # The upper bound of the space must be lower than our upper bound:
        o_high: np.ndarray = space.high
        if callable(self.high):
            if not self.high(o_high):
                return False
        elif self.high is not None:
            s_high = cast(Union[float, np.ndarray], self.high)
            if not np.less_equal(s_high, o_high).all():
                return False
        return True


D = TypeVar("D", bound=spaces.Discrete)


class Discretes(HigherSpace[D]):
    def __init__(
        self,
        n: Union[int, Predicate[int], Space[int]] = None,
        seed: int = None,
        dtype: Type[D] = spaces.Discrete,
    ):
        super().__init__(seed=seed, dtype=None)
        self.n = n
        self.dtype = dtype

    def sample(self) -> D:
        return self.dtype(n=np.random.randint(123))
        return super().sample()

    def contains(self, x: Any) -> bool:
        if not isinstance(x, self.dtype):
            return False
        if callable(self.n):
            if not self.n(x.n):
                return False
        return True


class SpaceWrapper(Space[S]):
    """Base class for wrapper around spaces."""

    def __init__(
        self,
        space: Space[S],
    ):
        super().__init__(shape=space.shape, dtype=space.dtype, seed=space.seed)
        self.space = space

    def __getattr__(self, attr: str):
        if attr.startswith("_"):
            raise AttributeError(f"Attempted to get missing private attribute '{attr}'")
        return getattr(self.space, attr)

    def seed(self, seed: Optional[int] = None):
        """Seed the rng of the wrapped space."""
        return self.space.seed(seed)

    def sample(self) -> S:
        """Take a sample from the wrapped space."""
        return self.space.sample()

    def contains(self, x) -> bool:
        """Checks if the wrapped space contains this sample."""
        return self.space.contains(x)

    def __str__(self):
        return f"<{type(self).__name__}{self.space}>"

    def __repr__(self) -> str:
        return str(self)

    def to_jsonable(self, sample_n):
        """Convert a batch of samples from this space to a JSONable data type."""
        return self.space.to_jsonable(sample_n)

    def from_jsonable(self, sample_n):
        """Convert a JSONable data type to a batch of samples from this space."""
        return self.space.from_jsonable(sample_n)


class SpaceWithAssumptions(SpaceWrapper[S]):
    """Adds assumptions (predicates) to a space, limiting the samples that are considered valid.

    ```python
    space: Space[Discrete] = Discretes().where(lambda n: n % 2 == 0)

    assert Discrete(2) in space
    assert Discrete(3) not in space
    ```
    """

    def __init__(self, space: Space[S], assumptions: List[Predicate[S]]):
        super().__init__(space=space)
        self.space = space
        self.assumptions = assumptions

    def contains(self, x) -> bool:
        """Checks if the wrapped space contains this sample and if it fits all assumptions."""
        return self.space.contains(x) and all(assumption(x) for assumption in self.assumptions)

    def where(self, *assumptions: Predicate[S]) -> SpaceWithAssumptions[S]:
        return type(self)(self.space, assumptions=self.assumptions + list(assumptions))

    def __repr__(self):
        return repr(self.space) + ".where(" + repr(self.assumptions) + ")"


class MatchesProperties(Generic[T]):
    """
    NOTE:
    - If a predicate for a given attribute is `None`, then it isn't checked.
    """

    def __init__(self, **attributes_to_predicates: Optional[Predicate]):
        self.attributes_to_predicates = attributes_to_predicates

    def __call__(self, x: T) -> bool:
        for attribute, predicate in self.attributes_to_predicates.items():
            if not hasattr(x, attribute):
                return False
            value = getattr(x, attribute)
            if predicate is None:
                continue
            assert callable(predicate), f"Predicate should be callables, not {predicate}"
            if not predicate(value):
                return False
        return True


E = TypeVar("E", bound=gym.Env)
from functools import singledispatch


@singledispatch
def fits_or_isin(
    space_or_predicate: Union[HigherSpace[S], Predicate[S], None], v: Optional[S]
) -> bool:
    if space_or_predicate is None:
        return True
    if isinstance(space_or_predicate, HigherSpace):
        return v in space_or_predicate
    assert callable(space_or_predicate)
    assert v is not None
    return space_or_predicate(v)


import gym.envs.registration
from simple_parsing.helpers import JsonSerializable
from dataclasses import dataclass

from simple_parsing.helpers.serialization.serializable import FrozenSerializable


@dataclass(frozen=True)
class SpacesSpec(FrozenSerializable):
    observation_space: Space
    action_space: Space
    reward_space: Optional[Space]


@fits_or_isin.register(SpacesSpec)
def _(space_spec: SpacesSpec, item: Union[gym.Env, SpacesSpec]) -> bool:
    return (
        fits_or_isin(space_spec.observation_space, item.observation_space)
        and fits_or_isin(space_spec.action_space, item.action_space)
        and fits_or_isin(space_spec.reward_space, getattr(item, "reward_space", None))
    )


spaces_registry: Dict[str, SpacesSpec] = {}


class EnvSpace(Space[E], MatchesProperties):
    def __init__(
        self,
        observation_space: Union[HigherSpace, Predicate[Space]] = None,
        action_space: Union[HigherSpace, Predicate[Space]] = None,
        reward_space: Union[HigherSpace, Predicate[Space]] = None,
        registry: EnvRegistry = gym.envs.registration.registry,
        seed: int = None,
    ):
        super().__init__(dtype=None, seed=seed)
        self.observation_space = observation_space
        self.action_space = action_space
        self.reward_space = reward_space
        self.registry = registry

    def sample(self) -> E:
        # IDEA: Could check within the registry, all the envs that match the assumptions, and then
        # sample one env at random?
        env_spec: gym.envs.registration.EnvSpec
        global spaces_registry
        eligible_environment_fns: Dict[str, gym.envs.registration.EnvSpec] = {}
        for env_id, env_spec in self.registry.env_specs.items():
            if env_id in spaces_registry:
                spaces = spaces_registry[env_id]
            else:
                # Create the env, which might be expensive.
                try:
                    env = env_spec.make()
                except Exception as exc:
                    print(f"Couldn't check spaces of env {env_id}: {exc}")
                    continue

                spaces = SpacesSpec(
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                    reward_space=getattr(env, "reward_space", None),
                )
                spaces_registry[env_id] = spaces
            if fits_or_isin(self._spaces, spaces):
                eligible_environment_fns[env_id] = env_spec
        env_id, env_spec = self.np_random.choice(list(eligible_environment_fns.items()))
        return env_spec.make()
        raise NotImplementedError

    @property
    def _spaces(self) -> SpacesSpec:
        return SpacesSpec(
            observation_space=self.observation_space,
            action_space=self.action_space,
            reward_space=self.reward_space,
        )

    def contains(self, x: E) -> bool:
        if not isinstance(x, gym.Env):
            return False
        obs_space = x.observation_space
        act_space = x.action_space
        rew_space = getattr(x, "reward_space", None)
        return all(
            fits_or_isin(self_space, x_space)
            for self_space, x_space in [
                (self.observation_space, obs_space),
                (self.action_space, act_space),
                (self.reward_space, rew_space),
            ]
        )
