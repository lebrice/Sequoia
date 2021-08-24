import inspect
from abc import ABC
from collections.abc import Sized
from functools import partial, singledispatch
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import gym
import numpy as np
from gym import spaces
from gym.envs import registry

from gym.envs.classic_control import (
    AcrobotEnv,
    CartPoleEnv,
    Continuous_MountainCarEnv,
    MountainCarEnv,
    PendulumEnv,
)
from gym.envs.registration import load, EnvSpec
from gym.vector.utils import batch_space
from gym.vector import VectorEnv
from torch.utils.data import DataLoader, IterableDataset

from sequoia.utils.logging_utils import get_logger

classic_control_envs = (
    AcrobotEnv,
    CartPoleEnv,
    PendulumEnv,
    MountainCarEnv,
    Continuous_MountainCarEnv,
)

classic_control_env_prefixes: Tuple[str, ...] = (
    "CartPole",
    "Pendulum",
    "Acrobot",
    "MountainCar",
    "MountainCarContinuous",
)


def is_classic_control_env(env: Union[str, gym.Env, Type[gym.Env]]) -> bool:
    """Returns `True` if the given env id, env class, or env instance is a
    classic-control env.

    Parameters
    ----------
    env : Union[str, gym.Env]
        Env id, or env class, or env instance.

    Returns
    -------
    bool
        Wether the given env is a classic-control env from Gym.

    Examples:

    >>> import gym
    >>> is_classic_control_env("CartPole-v0")
    True
    >>> is_classic_control_env("Breakout-v1")
    False
    >>> is_classic_control_env("bob")
    False
    >>> from gym.envs.classic_control import CartPoleEnv
    >>> is_classic_control_env(CartPoleEnv)
    True
    """
    if isinstance(env, partial):
        if env.func is gym.make and isinstance(env.args[0], str):
            logger.warning(
                RuntimeWarning(
                    "Don't pass partial(gym.make, 'some_env'), just use the env string instead."
                )
            )
            env = env.args[0]
    if isinstance(env, str):
        try:
            spec = registry.spec(env)
            if isinstance(spec.entry_point, str):
                return "gym.envs.classic_control" in spec.entry_point
            if inspect.isclass(spec.entry_point):
                env = spec.entry_point
        except gym.error.Error as e:
            # malformed env id, for instance.
            logger.debug(f"can't tell if env id {env} is a classic-control env! ({e})")
            return False

    if inspect.isclass(env):
        return issubclass(env, classic_control_envs)
    if isinstance(env, gym.Env):
        return isinstance(env.unwrapped, classic_control_envs)
    return False


def is_proxy_to(
    env, env_type_or_types: Union[Type[gym.Env], Tuple[Type[gym.Env], ...]]
) -> bool:
    """Returns wether `env` is a proxy to an env of the given type or types."""
    from sequoia.client.env_proxy import EnvironmentProxy

    return isinstance(env.unwrapped, EnvironmentProxy) and issubclass(
        env.unwrapped._environment_type, env_type_or_types
    )


def is_atari_env(env: Union[str, gym.Env]) -> bool:
    """Returns `True` if the given env id, env class, or env instance is a
    Atari environment.

    Parameters
    ----------
    env : Union[str, gym.Env]
        Env id, or env class, or env instance.

    Returns
    -------
    bool
        Wether the given env is an Atari env from Gym.

    Examples:

    >>> import gym
    >>> is_atari_env("CartPole-v0")
    False
    >>> is_atari_env("Breakout-v0")
    True
    >>> is_atari_env("bob")
    False
    >>> from gym.envs.atari import AtariEnv  # requires atari_py to be installed
    >>> is_atari_env(AtariEnv)
    True
    """
    # TODO: Add more names from the atari environments, or figure out a smarter
    # way to do this.
    if isinstance(env, partial):
        if env.func is gym.make and isinstance(env.args[0], str):
            logger.warning(
                RuntimeWarning(
                    "Don't pass partial(gym.make, 'some_env'), just use the env string instead."
                )
            )
            env = env.args[0]
    # assert False, [env_spec for env_spec in registry.all()]
    if isinstance(env, str):  # and env.startswith("Breakout"):
        try:
            spec = registry.spec(env)
            if isinstance(spec.entry_point, str):
                return "gym.envs.atari" in spec.entry_point
            if inspect.isclass(spec.entry_point):
                env = spec.entry_point
        except gym.error.Error as e:
            # malformed env id, for instance.
            logger.debug(f"can't tell if env id {env} is an atari env! ({e})")
            return False

    try:
        from gym.envs.atari import AtariEnv

        if inspect.isclass(env) and issubclass(env, AtariEnv):
            return True
        return isinstance(env, gym.Env) and isinstance(env.unwrapped, AtariEnv)
    except gym.error.DependencyNotInstalled:
        return False
    return False


def get_env_class(
    env: Union[str, gym.Env, Type[gym.Env], Callable[[], gym.Env]]
) -> Type[gym.Env]:
    if isinstance(env, partial):
        if env.func is gym.make and isinstance(env.args[0], str):
            return get_env_class(env.args[0])
        return get_env_class(env.func)
    if isinstance(env, str):
        return load(env)
    if isinstance(env, gym.Wrapper):
        return type(env.unwrapped)
    if isinstance(env, gym.Env):
        return type(env)
    if inspect.isclass(env) and issubclass(env, gym.Env):
        return env
    raise NotImplementedError(
        f"Don't know how to get the class of env being used by {env}!"
    )


def is_monsterkong_env(env: Union[str, gym.Env, Callable[[], gym.Env]]) -> bool:
    if isinstance(env, str):
        return env.lower().startswith(("metamonsterkong", "monsterkong"))
    try:
        from meta_monsterkong.make_env import MetaMonsterKongEnv

        if inspect.isclass(env):
            return issubclass(env, MetaMonsterKongEnv)
        if isinstance(env, gym.Env):
            return isinstance(env, MetaMonsterKongEnv)
        return False
    except ImportError:
        return False


logger = get_logger(__file__)

EnvType = TypeVar("EnvType", bound=gym.Env)
ObservationType = TypeVar("ObservationType")
ActionType = TypeVar("ActionType")
RewardType = TypeVar("RewardType")


class StepResult(NamedTuple):
    observation: ObservationType
    reward: RewardType
    done: Union[bool, Sequence[bool]]
    info: Union[Dict, Sequence[Dict]]


def has_wrapper(
    env: gym.Wrapper,
    wrapper_type_or_types: Union[Type[gym.Wrapper], Tuple[Type[gym.Wrapper], ...]],
) -> bool:
    """Returns wether the given `env` has a wrapper of type `wrapper_type`.

    Args:
        env (gym.Wrapper): a gym.Wrapper or a gym environment.
        wrapper_type (Type[gym.Wrapper]): A type of Wrapper to check for.

    Returns:
        bool: Wether there is a wrapper of that type wrapping `env`.
    """
    # avoid cycles, although that would be very weird to encounter.
    while hasattr(env, "env") and env.env is not env:
        if isinstance(env, wrapper_type_or_types):
            return True
        env = env.env
    return isinstance(env, wrapper_type_or_types)


class MayCloseEarly(gym.Wrapper, ABC):
    """ABC for Wrappers that may close an environment early depending on some
    conditions.

    NOTE: Raises a gym.error.ClosedEnvironmentError when calling `step` and `reset` when
    the env is closed.
    """

    def __init__(self, env: gym.Env, **kwargs):
        super().__init__(env, **kwargs)
        self._is_closed: bool = False

    def is_closed(self) -> bool:
        # First, make sure that we're not 'overriding' the 'is_closed' of the
        # wrapped environment.
        if hasattr(self.env, "is_closed"):
            assert callable(self.env.is_closed)
            self._is_closed = self.env.is_closed()
        return self._is_closed

    def closed_error_message(self) -> str:
        """Return the error message to use when attempting to use the closed env.

        This can be useful for wrappers that close when a given condition is reached,
        e.g. a number of episodes has been performed, which could return a more relevant
        message here.
        """
        return "Env is closed"

    def reset(self, **kwargs):
        if self.is_closed():
            raise gym.error.ClosedEnvironmentError(
                f"Can't call `reset()`: {self.closed_error_message()}"
            )
        return super().reset(**kwargs)

    def step(self, action):
        if self.is_closed():
            raise gym.error.ClosedEnvironmentError(
                f"Can't call `step()`: {self.closed_error_message()}"
            )
        return super().step(action)

    def close(self) -> None:
        if self.is_closed():
            # TODO: Prevent closing an environment twice?
            return
            # raise gym.error.ClosedEnvironmentError(self.closed_error_message())
        self.env.close()
        self._is_closed = True


from .env_dataset import EnvDataset
from .policy_env import PolicyEnv
from .iterable_wrapper import IterableWrapper

class RenderEnvWrapper(IterableWrapper):
    """Simple Wrapper that renders the env at each step."""

    def __init__(self, env: gym.Env, display: Any = None):
        super().__init__(env)
        # TODO: Maybe use the given display?

    def step(self, action):
        self.env.render("human")
        return self.env.step(action)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
