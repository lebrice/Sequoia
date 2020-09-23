"""Creates an IterableDataset from a gym env by applying different wrappers.
"""
import copy
from functools import partial, reduce
from typing import Callable, Dict, Iterable, List, Tuple, Type, TypeVar, Union

import gym
from gym import Wrapper
from gym.envs.classic_control import CartPoleEnv

from common.gym_wrappers import *
from utils.logging_utils import get_logger

logger = get_logger(__file__)

W = TypeVar("W", bound=Union[gym.Env, gym.Wrapper])

WrapperAndKwargs = Tuple[Type[gym.Wrapper], Dict]
# TODO: @lebrice Not sure if this is a good idea, or that convenient, but
# we could use a dict like this to save the 'default' wrappers to use before and
# after batching for each env name or type of env.
# TODO: Figure out the right ordering to use for the wrappers.
default_wrappers_for_env: Dict[str, Iterable[WrapperAndKwargs]] = {
    "CartPole-v0": [ConvertToFromTensors],
}
default_post_batch_wrappers_for_env: Dict[str, Iterable[WrapperAndKwargs]] = {
    "CartPole-v0": [],
}
import gym
from gym.vector import make, AsyncVectorEnv, SyncVectorEnv


def make_batched_env(
        base_env: Union[str, Callable[[], gym.Env]] = "CartPole-v0",
        batch_size: int = 10,
        pre_batch_wrappers: Iterable[Tuple[Type[gym.Wrapper], Dict]] = None,
        post_batch_wrappers: Iterable[Tuple[Type[gym.Wrapper], Dict]] = None,
        use_default_wrappers_for_env: bool = True,
    ) -> Union[BatchEnv, EnvDataset]:
    """Creates a batched env, applying wrappers before and after the batching.

    Args:
        base_env (Union[str, Callable[[], gym.Env]], optional): [description]. Defaults to "CartPole-v0".
        batch_size (int, optional): [description]. Defaults to 10.
        pre_batch_wrappers (Iterable[Tuple[Type[gym.Wrapper], Dict]], optional): [description]. Defaults to None.
        post_batch_wrappers (Iterable[Tuple[Type[gym.Wrapper], Dict]], optional): [description]. Defaults to None.
        use_default_wrappers_for_env (bool, optional): [description]. Defaults to True.

    Raises:
        RuntimeError: [description]
        RuntimeError: [description]

    Returns:
        Union[BatchEnv, EnvDataset]: [description]
    """
    if not pre_batch_wrappers and use_default_wrappers_for_env:
        if hasattr(base_env, "spec"):
            assert False, base_env.spec
        pre_batch_wrappers = default_wrappers_for_env.get(base_env, [])
    if not post_batch_wrappers and use_default_wrappers_for_env:
        post_batch_wrappers = default_post_batch_wrappers_for_env.get(base_env, [])
    
    pre_batch_wrapper_fns = _make_wrapper_fns(pre_batch_wrappers)
    post_batch_wrapper_fns = _make_wrapper_fns(post_batch_wrappers)

    if isinstance(base_env, str):
        base_env_factory = partial(gym.make, base_env)

    elif callable(base_env):
        base_env_factory = base_env

    elif isinstance(base_env, gym.Env):
        # TODO: Check that there isn't a smarter way of doing this, maybe
        # by getting the env spec and creating a new one like it using
        # `gym.make`?
        logger.warning(RuntimeWarning(
            f"Will try to use deepcopy as an env factory.. but this is really "
            f"less than ideal!"))
        logger.debug(f"Env spec: {env.spec}")
        base_env_factory = partial(copy.deepcopy, base_env)
    
    else:
        raise NotImplementedError(
            f"Unsupported base env: {base_env}. Must be "
            f"either a string, a gym env, or a callable for now."
        )
    
    def pre_batch_env_factory():
        env = base_env_factory()
        for wrapper_fn in pre_batch_wrapper_fns:
            env = wrapper_fn(env)
        return env
    
    batched_env = BatchEnv(
        env_factory=pre_batch_env_factory,
        batch_size=batch_size,
    )
    # apply all the post-batch wrappers to the batched env:
    env = reduce(
        lambda wrapper_fn, wrapped: wrapper_fn(wrapped),
        post_batch_wrapper_fns,
        batched_env,
    )
    return env


def _make_wrapper_fns(
    wrappers_and_args: Iterable[Union[Type[Wrapper],
                                Tuple[Type[Wrapper], Dict]]]
                                ) -> List[Callable[[Wrapper], Wrapper]]:
    """ Given a list of either wrapper classes or (wrapper, kwargs) tuples,
    returns a list of callables, each of which just takes an env and wraps
    it using the wrapper and the kwargs, if present.
    """
    wrappers_and_args = list(wrappers_and_args or [])
    wrapper_functions: List[Callable[[gym.Wrapper], gym.Wrapper]] = []
    for wrapper_and_args in wrappers_and_args:
        if isinstance(wrapper_and_args, (tuple, list)):
            # List element was a tuple with (wrapper, (args?), kwargs).
            wrapper, *args, kwargs = wrapper_and_args
            logger.debug(f"Wrapper: {wrapper}, args: {args}, kwargs: {kwargs}") 
            wrapper_fn = partial(wrapper, *args, **kwargs)
        else:
            # list element is a type of Wrapper or some kind of callable.
            wrapper_fn = wrapper_and_args
        wrapper_functions.append(wrapper_fn)
    return wrapper_functions
