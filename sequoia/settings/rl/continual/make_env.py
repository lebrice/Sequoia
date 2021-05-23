"""Creates an IterableDataset from a gym env by applying different wrappers.
"""
import multiprocessing as mp
import warnings
from functools import partial
from typing import (Callable, Dict, Iterable, List, Optional, Tuple, Type,
                    TypeVar, Union)

import gym
from gym import Wrapper
from gym.envs.classic_control import CartPoleEnv
from gym.vector import VectorEnv

from sequoia.common.gym_wrappers import ConvertToFromTensors
from sequoia.common.gym_wrappers.batch_env import (AsyncVectorEnv,
                                                   BatchedVectorEnv,
                                                   SyncVectorEnv)
from sequoia.common.spaces import Sparse
from sequoia.utils.logging_utils import get_logger

logger = get_logger(__file__)

W = TypeVar("W", bound=Union[gym.Env, gym.Wrapper])

WrapperAndKwargs = Tuple[Type[gym.Wrapper], Dict]


def make_batched_env(base_env: Union[str, Callable],
                     batch_size: int = 10,
                     wrappers: Iterable[Union[Type[Wrapper], WrapperAndKwargs]] = None,
                     shared_memory: bool = True,
                     num_workers: Optional[int] = None,
                     **kwargs) -> VectorEnv:
    """Create a vectorized environment from multiple copies of an environment.

    NOTE: This function does pretty much the same as `gym.vector.make`, but with
    a bit more flexibility:
    - Allows passing an env factory to start with, rather than only taking ids.
    - Allows passing wrappers to be added to the env on
        each worker, as well as wrappers to add on top of the returned (batched) env.
    - Allows passing tuples of (Type[Wrapper, ])

    Parameters
    ----------
    base_env : str
        The environment ID (or an environment factory). This must be a valid ID
        from the registry.

    batch_size : int
        Number of copies of the environment (as well as batch size). 
    
    num_workers : Optional[int]
        Number of workers to use. When `None` (default), uses as many workers as
        there are CPUs on this machine. When 0, the returned environment will be
        a `SyncVectorEnv`. When `num_workers` == `batch_size`, returns an
        AsyncVectorEnv. When `num_workers` != `batch_size`, returns a
        `BatchVectorEnv`.

    wrappers : Callable or Iterable of Callables (default: `None`)
        If not `None`, then apply the wrappers to each internal environment
        during creation.
    
    **kwargs : Dict
        Keyword arguments to be passed to `gym.make` when `base_env` is an id.

    Returns
    -------
    env : `gym.vector.VectorEnv` instance
        The vectorized environment.

    Example
    -------
    >>> import gym
    >>> env = gym.vector.make('CartPole-v1', 3)
    >>> env.seed([123, 456, 789])
    >>> env.reset()
    array([[ 0.02078762, -0.01301236, -0.0209893 , -0.03935255],
           [ 0.03271029, -0.01839286,  0.00746923,  0.0193136 ],
           [ 0.01767251,  0.00792448,  0.02225722, -0.03434491]],
          dtype=float32)
    """
    # Get the default wrappers, if needed.
    wrappers = wrappers or []
    
    base_env_factory: Callable[[], gym.Env]
    if isinstance(base_env, str):
        base_env_factory = partial(gym.make, base_env)
    elif callable(base_env):
        base_env_factory = base_env
    else:
        raise NotImplementedError(
            f"Unsupported base env: {base_env}. Must be "
            f"either a string or a callable for now."
        )

    def pre_batch_env_factory():
        env = base_env_factory(**kwargs)
        for wrapper in wrappers:
            if isinstance(wrapper, tuple):
                assert len(wrapper) == 2 and isinstance(wrapper[1], dict)
                wrapper = partial(wrapper[0], **wrapper[1])
            env = wrapper(env)
        return env

    if batch_size is None:
        return pre_batch_env_factory()
    
    env_fns = [pre_batch_env_factory for _ in range(batch_size)]

    if num_workers is None:
        if batch_size == 1:
            num_workers = 0
        else:
            num_workers = min(mp.cpu_count(), batch_size)

    if num_workers == 0:
        if batch_size > 1:
            warnings.warn(UserWarning(
                f"Running {batch_size} environments in series, which might be "
                f"slow. Consider setting the `num_workers` argument, perhaps to "
                f"the number of CPUs on your machine."
            ))
        return SyncVectorEnv(env_fns)
    
    if num_workers == batch_size:
        return AsyncVectorEnv(env_fns, shared_memory=shared_memory)
    
    return BatchedVectorEnv(env_fns, shared_memory=shared_memory, n_workers=num_workers)

   


def wrap(env: gym.Env,
         wrappers: Iterable[Union[Type[Wrapper], WrapperAndKwargs]]) -> Wrapper:
    wrappers = list(wrappers)
    # Convert the list of wrapper types or (wrapper_type, kwargs) tuples into
    # a list of callables that we can apply successively to the env.
    wrapper_fns = _make_wrapper_fns(wrappers)
    for wrapper_fn in wrapper_fns:
        env = wrapper_fn(env)
    return env

def _make_wrapper_fns(wrappers_and_args: Iterable[Union[Type[Wrapper],
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
