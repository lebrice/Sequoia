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
    """ Returns wether `env` is a proxy to an env of the given type or types.
    """
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




def get_env_class(env: Union[str, gym.Env, Type[gym.Env], Callable[[], gym.Env]]) -> Type[gym.Env]:
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


class StepResult(NamedTuple, Generic[ObservationType, RewardType]):
    state: ObservationType
    reward: RewardType
    done: bool
    info: Dict


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
    """ ABC for Wrappers that may close an environment early depending on some
    conditions.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._is_closed: bool = False

    def is_closed(self) -> bool:
        # First, make sure that we're not 'overriding' the 'is_closed' of the
        # wrapped environment.
        if hasattr(self.env, "is_closed"):
            assert callable(self.env.is_closed)
            self._is_closed = self.env.is_closed()
        return self._is_closed

    def close(self) -> None:
        self.env.close()
        self._is_closed = True


from .env_dataset import EnvDataset
from .policy_env import PolicyEnv


class IterableWrapper(MayCloseEarly, IterableDataset, Generic[EnvType], ABC):
    """ ABC for a gym Wrapper that supports iterating over the environment.

    This allows us to wrap dataloader-based Environments and still use the gym
    wrapper conventions, as well as iterate over a gym environment as in the
    Active-dataloader case.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def __next__(self):
        # TODO: This is tricky. We want the wrapped env to use *our* step,
        # reset(), action(), observation(), reward() methods, instead of its own!
        # Otherwise if we are transforming observations for example, those won't
        # be affected.
        # logger.debug(f"Wrapped env {self.env} isnt a PolicyEnv or an EnvDataset")
        # return type(self.env).__next__(self)
        from sequoia.settings.rl.environment import ActiveDataLoader
        from sequoia.settings.sl.environment import PassiveEnvironment
        if has_wrapper(self.env, EnvDataset) or is_proxy_to(
            self.env, (EnvDataset, ActiveDataLoader)
        ):
            obs, reward, done, info = self.step(self.action_)
            return obs
            # raise RuntimeError(f"WIP: Dropping this '__next__' API in RL.")
            # logger.debug(f"Wrapped env is an EnvDataset, using EnvDataset.__iter__.")
            # return EnvDataset.__next__(self)
            # return EnvDataset.__next__(self)
        return self.env.__next__()
        # return self.observation(obs)

    def observation(self, observation):
        logger.debug(f"Observation won't be transformed.")
        return observation

    def action(self, action):
        return action

    def reward(self, reward):
        return reward

    def __len__(self):
        return self.env.__len__()

    def send(self, action):
        # TODO: Make `send` use `self.step`, that way wrappers can apply the same way to
        # RL and SL environments.
        if hasattr(self.unwrapped, "send"):
            action = self.action(action)
            reward = self.env.send(action)
            reward = self.reward(reward)
            return reward
        else:
                
            self.action_ = action
            self.observation_, self.reward_, self.done_, self.info_ = self.step(action)
            return self.reward_

        # (Option 1 below)
        # return self.env.send(action)
        # (Option 2 below)
        # return self.env.send(self.action(action))

        # (Option 3 below)
        # return type(self.env).send(self, action)

        # (Following option 4 below)
        if has_wrapper(self.env, EnvDataset):
            # logger.debug(f"Wrapped env is an EnvDataset, using EnvDataset.send.")
            return EnvDataset.send(self, action)

        if hasattr(self.env, "send"):
            action = self.action(action)
            reward = self.env.send(action)
            reward = self.reward(reward)
            return reward

    def __iter__(self) -> Iterator:
        # Option 1: Return the iterator from the wrapped env. This ignores
        # everything in the wrapper.
        # return self.env.__iter__()

        # Option 2: apply the transformations on the items yielded by the
        # iterator of the wrapped env (this doesn't use the self.observaion(), self.action())
        # from .transform_wrappers import TransformObservation, TransformAction, TransformReward
        # return map(self.observation, self.env.__iter__())

        # Option 3: Calling the method on the wrapped env, but with `self` being
        # the wrapper, rather than the wrapped env:
        # return type(self.env).__iter__(self)

        # Option 4: Slight variation on option 3: We cut straight to the
        # EnvDataset iterator.

        from sequoia.settings.rl.environment import ActiveDataLoader
        from sequoia.settings.sl.environment import PassiveEnvironment

        if has_wrapper(self.env, EnvDataset) or is_proxy_to(
            self.env, (EnvDataset, ActiveDataLoader)
        ):
            # logger.debug(f"Wrapped env is an EnvDataset, using EnvDataset.__iter__ with the wrapper as `self`.")
            return EnvDataset.__iter__(self)

        if has_wrapper(self.env, PolicyEnv) or is_proxy_to(self.env, PolicyEnv):
            # logger.debug(f"Wrapped env is a PolicyEnv, will use PolicyEnv.__iter__ with the wrapper as `self`.")
            return PolicyEnv.__iter__(self)

        # NOTE: This works even though IterableDataset isn't a gym.Wrapper.
        if not has_wrapper(self.env, IterableDataset) and not isinstance(
            self.env, DataLoader
        ):
            logger.warning(
                UserWarning(
                    f"Will try to iterate on a wrapper for env {self.env} which "
                    f"doesn't have the EnvDataset or PolicyEnv wrappers and isn't "
                    f"an IterableDataset."
                )
            )

        return self.env.__iter__()

    @property
    def wrapping_passive_env(self) -> bool:
        """ Returns wether this wrapper is applied over a 'passive' env, in which case
        iterating over the env will yield (up to) 2 items, rather than just 1.
        """
        from sequoia.settings.sl.environment import PassiveEnvironment

        return isinstance(self.unwrapped, PassiveEnvironment) or is_proxy_to(
            self, PassiveEnvironment
        )

    def __setattr__(self, attr, value):
        """ Redirect the __setattr__ of attributes 'owned' by the EnvDataset to
        the EnvDataset.

        We need to do this because we change the value of `self` and call
        EnvDataset.__iter__(self), which might get and set attributes to/from
        `self`, which is what you'd expect normally. However when `self` is a
        wrapper over the env, rather than the env itself, then when attributes
        are set on `self` inside __iter__ or __next__ or send, etc, they are
        actually set on the wrapper, rather than on the env.

        We solve this by detecting when an attribute with a name ending with "_"
        and part of a given list of attributes is set.
        """
        if attr.endswith("_") and has_wrapper(self.env, EnvDataset):
            if attr in {
                "observation_",
                "action_",
                "reward_",
                "done_",
                "info_",
                "n_sends_",
            }:
                # logger.debug(f"Attribute {attr} will be set on the wrapped env rather than on the wrapper itself.")
                env = self.env
                while not isinstance(env, EnvDataset) and env.env is not env:
                    env = env.env
                assert isinstance(env, EnvDataset)
                setattr(env, attr, value)
        else:
            object.__setattr__(self, attr, value)


class RenderEnvWrapper(IterableWrapper):
    """ Simple Wrapper that renders the env at each step. """

    def __init__(self, env: gym.Env, display: Any = None):
        super().__init__(env)
        # TODO: Maybe use the given display?

    def step(self, action):
        self.env.render("human")
        return self.env.step(action)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
