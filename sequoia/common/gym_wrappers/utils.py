import inspect
from abc import ABC
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)
import warnings

import gym
import numpy as np
from gym.envs import registry
from gym.envs.classic_control import (
    AcrobotEnv,
    CartPoleEnv,
    Continuous_MountainCarEnv,
    MountainCarEnv,
    PendulumEnv,
)
from gym.envs.registration import load
from gym.vector import VectorEnv
from torch.utils.data import IterableDataset

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
logger = get_logger(__name__)


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


def is_proxy_to(env, env_type_or_types: Union[Type[gym.Env], Tuple[Type[gym.Env], ...]]) -> bool:
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
    >>> is_atari_env("bob")
    False
    >>> # is_atari_env("ALE/Breakout-v5")
    # True
    >>> # is_atari_env("Breakout-v0")
    # True

    NOTE: Removing this doctest, since recent changes to gym have changed this a bit.
    >>> #from gym.envs import atari
    >>> #is_atari_env(atari.AtariEnv) # requires atari_py to be installed
    # True
    """
    from sequoia.settings.rl.envs import ATARI_PY_INSTALLED

    if not isinstance(env, (str, gym.Env)):
        raise RuntimeError(f"`env` needs to be either a str or gym env, not {env}")
    if isinstance(env, str):
        try:
            spec = registry.spec(env)
        except gym.error.NameNotFound:
            return False
        except gym.error.NamespaceNotFound:
            return False
        if spec.namespace is None:
            return False
        return spec.namespace is "ALE"
    if not ATARI_PY_INSTALLED:
        return False
    raise NotImplementedError(f"TODO: Check if isinstance(env.unwrapped, AtariEnv)")

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
                return "gym.envs.atari" in spec.entry_point or "ale_py" in spec.entry_point
            if inspect.isclass(spec.entry_point):
                env = spec.entry_point
        except gym.error.Error as e:
            # malformed env id, for instance.
            logger.debug(f"can't tell if env id {env} is an atari env! ({e})")
            return False

    try:
        from gym.envs import atari

        AtariEnv = atari.AtariEnv
        if inspect.isclass(env) and issubclass(env, AtariEnv):
            return True
        return isinstance(env, gym.Env) and isinstance(env.unwrapped, AtariEnv)
    except (ImportError, gym.error.DependencyNotInstalled):
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
    raise NotImplementedError(f"Don't know how to get the class of env being used by {env}!")


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

    WIP: Also prevents calling `step` and `reset` on a closed env.
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


class IterableWrapper(MayCloseEarly, IterableDataset, Generic[EnvType], ABC):
    """ABC for a gym Wrapper that supports iterating over the environment.

    This allows us to wrap dataloader-based Environments and still use the gym
    wrapper conventions, as well as iterate over a gym environment as in the
    Active-dataloader case.

    NOTE: We have IterableDataset as a base class here so that we can pass a wrapped env
    to the DataLoader function. This wrapper however doesn't perform the actual
    iteration, and instead depends on the wrapped environment already supporting
    iteration.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        from sequoia.settings.sl import PassiveEnvironment

        self.wrapping_passive_env = isinstance(self.unwrapped, PassiveEnvironment)

    @property
    def is_vectorized(self) -> bool:
        """Returns wether this wrapper is wrapping a vectorized environment."""
        return isinstance(self.unwrapped, VectorEnv)

    def __next__(self):
        # TODO: This is tricky. We want the wrapped env to use *our* step,
        # reset(), action(), observation(), reward() methods, instead of its own!
        # Otherwise if we are transforming observations for example, those won't
        # be affected.
        # logger.debug(f"Wrapped env {self.env} isnt a PolicyEnv or an EnvDataset")
        # return type(self.env).__next__(self)
        from sequoia.settings.rl.environment import ActiveDataLoader

        # from sequoia.settings.sl.environment import PassiveEnvironment

        if has_wrapper(self.env, EnvDataset) or is_proxy_to(
            self.env, (EnvDataset, ActiveDataLoader)
        ):
            obs, reward, done, info = self.step(self.unwrapped.action_)
            return obs
            # raise RuntimeError(f"WIP: Dropping this '__next__' API in RL.")
            # logger.debug(f"Wrapped env is an EnvDataset, using EnvDataset.__iter__.")
            # return EnvDataset.__next__(self)
            # return EnvDataset.__next__(self)
        return self.env.__next__()
        # return self.observation(obs)

    def observation(self, observation):
        # logger.debug(f"Observation won't be transformed.")
        return observation

    def action(self, action):
        return action

    def reward(self, reward):
        return reward

    # def __len__(self):
    #     return self.env.__len__()

    def get_length(self) -> Optional[int]:
        """Attempts to return the "length" (in number of steps/batches) of this env.

        When not possible, returns None.

        NOTE: This is a bit ugly, but the idea seems alright.
        """
        try:
            # Try to call self.__len__() without recursing into the wrapped env:
            return len(self)
        except TypeError:
            pass
        try:
            # Try to call self.env.__len__() without recursing into the wrapped^2 env:
            return len(self.env)
        except TypeError:
            pass
        try:
            # Try to call self.env.__len__(), allowing recursing down the chain:
            return self.env.__len__()
        except TypeError:
            pass
        try:
            # If all else fails, delegate to the wrapped env's length() method, if any:
            return self.env.get_length()
        except AttributeError:
            pass
        # In the worst case, return None, meaning that we don't have a length.
        return None

    def send(self, action):
        # TODO: Make `send` use `self.step`, that way wrappers can apply the same way to
        # RL and SL environments.
        if self.wrapping_passive_env:
            action = self.action(action)
            reward = self.env.send(action)
            reward = self.reward(reward)
            return reward

        self.unwrapped.action_ = action
        (
            self.unwrapped.observation_,
            self.unwrapped.reward_,
            self.unwrapped.done_,
            self.unwrapped.info_,
        ) = self.step(action)
        return self.unwrapped.reward_

        # (Option 1 below)
        # return self.env.send(action)
        # (Option 2 below)
        # return self.env.send(self.action(action))

        # (Option 3 below)
        # return type(self.env).send(self, action)

        # (Following option 4 below)
        # if has_wrapper(self.env, EnvDataset):
        #     # logger.debug(f"Wrapped env is an EnvDataset, using EnvDataset.send.")
        #     return EnvDataset.send(self, action)

        # if hasattr(self.env, "send"):
        #     action = self.action(action)
        #     reward = self.env.send(action)
        #     reward = self.reward(reward)
        #     return reward

    def __iter__(self) -> Iterator:
        # TODO: Pretty sure this could be greatly simplified by just always using the loop from EnvDataset.
        if self.wrapping_passive_env:
            # NOTE: Also applies the `self.observation` `self.reward` methods while
            # iterating.
            for obs, rewards in self.env:
                obs = self.observation(obs)
                if rewards is not None:
                    rewards = self.reward(rewards)
                yield obs, rewards
        else:
            self.unwrapped.observation_ = self.reset()
            self.unwrapped.done_ = False
            self.unwrapped.action_ = None
            self.unwrapped.reward_ = None

            # Yield the first observation_.
            yield self.unwrapped.observation_

            if self.unwrapped.action_ is None:
                raise RuntimeError(
                    f"You have to send an action using send() between every "
                    f"observation. (env = {self})"
                )

            def done_is_true(done: Union[bool, np.ndarray, Sequence[bool]]) -> bool:
                return done if isinstance(done, bool) or not done.shape else all(done)

            while not any([done_is_true(self.unwrapped.done_), self.is_closed()]):
                # logger.debug(f"step {self.n_steps_}/{self.max_steps},  (episode {self.n_episodes_})")

                # Set those to None to force the user to call .send()
                self.unwrapped.action_ = None
                self.unwrapped.reward_ = None
                yield self.unwrapped.observation_

                if self.unwrapped.action_ is None:
                    raise RuntimeError(
                        f"You have to send an action using send() between every "
                        f"observation. (env = {self})"
                    )

        # assert False, "WIP"

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

        # from sequoia.settings.rl.environment import ActiveDataLoader
        # from sequoia.settings.sl.environment import PassiveEnvironment

        # if has_wrapper(self.env, EnvDataset) or is_proxy_to(
        #     self.env, (EnvDataset, ActiveDataLoader)
        # ):
        #     # logger.debug(f"Wrapped env is an EnvDataset, using EnvDataset.__iter__ with the wrapper as `self`.")
        #     return EnvDataset.__iter__(self)

        # # TODO: Should probably remove this since we don't actually use this 'PolicyEnv'.
        # if has_wrapper(self.env, PolicyEnv) or is_proxy_to(self.env, PolicyEnv):
        #     # logger.debug(f"Wrapped env is a PolicyEnv, will use PolicyEnv.__iter__ with the wrapper as `self`.")
        #     return PolicyEnv.__iter__(self)

        # # NOTE: This works even though IterableDataset isn't a gym.Wrapper.
        # if not has_wrapper(self.env, IterableDataset) and not isinstance(
        #     self.env, DataLoader
        # ):
        #     logger.warning(
        #         UserWarning(
        #             f"Will try to iterate on a wrapper for env {self.env} which "
        #             f"doesn't have the EnvDataset or PolicyEnv wrappers and isn't "
        #             f"an IterableDataset."
        #         )
        #     )
        # # if isinstance(self.env, DataLoader):
        # #     return self.env.__iter__()
        # # raise NotImplementedError(f"Wrapper {self} doesn't know how to iterate on {self.env}.")
        # return self.env.__iter__()

    # @property
    # def wrapping_passive_env(self) -> bool:
    #     """ Returns wether this wrapper is applied over a 'passive' env, in which case
    #     iterating over the env will yield (up to) 2 items, rather than just 1.
    #     """
    #     from sequoia.settings.sl.environment import PassiveEnvironment

    #     return isinstance(self.unwrapped, PassiveEnvironment) or is_proxy_to(
    #         self, PassiveEnvironment
    #     )

    # def __setattr__(self, attr, value):
    #     """
    #     TODO: Remove/replace this:

    #     Redirect the __setattr__ of attributes 'owned' by the EnvDataset to
    #     the EnvDataset.

    #     We need to do this because we change the value of `self` and call
    #     EnvDataset.__iter__(self), which might get and set attributes to/from
    #     `self`, which is what you'd expect normally. However when `self` is a
    #     wrapper over the env, rather than the env itself, then when attributes
    #     are set on `self` inside __iter__ or __next__ or send, etc, they are
    #     actually set on the wrapper, rather than on the env.

    #     We solve this by detecting when an attribute with a name ending with "_"
    #     and part of a given list of attributes is set.
    #     """
    #     if attr.endswith("_") and has_wrapper(self.env, EnvDataset):
    #         if attr in {
    #             "observation_",
    #             "action_",
    #             "reward_",
    #             "done_",
    #             "info_",
    #             "n_sends_",
    #         }:
    #             # logger.debug(f"Attribute {attr} will be set on the wrapped env rather than on the wrapper itself.")
    #             env = self.env
    #             while not isinstance(env, EnvDataset) and env.env is not env:
    #                 env = env.env
    #             assert isinstance(env, EnvDataset)
    #             setattr(env, attr, value)
    #     else:
    #         object.__setattr__(self, attr, value)


class RenderEnvWrapper(IterableWrapper):
    """Simple Wrapper that renders the env at each step."""

    def __init__(self, env: gym.Env, display: Any = None):
        super().__init__(env)
        # TODO: Maybe use the given display?

    def step(self, action):
        self.env.render("human")
        return self.env.step(action)


def tile_images(img_nhwc):
    """
    TAKEN FROM https://github.com/openai/gym/pull/1624/files

    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.
    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)

    N, h, w, c = img_nhwc.shape
    if c not in {1, 3}:
        img_nhwc = img_nhwc.transpose([0, 2, 3, 1])
        N, h, w, c = img_nhwc.shape
    assert c in {1, 3}

    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N) / H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0] * 0 for _ in range(N, H * W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H * h, W * w, c)
    return img_Hh_Ww_c


if __name__ == "__main__":
    import doctest

    doctest.testmod()
