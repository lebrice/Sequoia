"""Creates an IterableDataset from a gym env by applying different wrappers.
"""
import copy
import multiprocessing as mp
from functools import partial
from typing import Callable, Dict, Iterable, List, Tuple, Type, TypeVar, Union

import gym
from gym import Wrapper
from gym.envs.classic_control import CartPoleEnv
from gym.vector import VectorEnv
from utils.logging_utils import get_logger
from common.gym_wrappers.sparse_space import Sparse
from common.gym_wrappers.batch_env import AsyncVectorEnv, BatchedVectorEnv, SyncVectorEnv
from common.gym_wrappers import ConvertToFromTensors
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

def make_batched_env(base_env: Union[str, Callable],
                     batch_size: int = 10,
                     wrappers: Iterable[Union[Type[Wrapper], WrapperAndKwargs]] = None,
                     use_default_wrappers_for_env: bool = True,
                     asynchronous: bool = True,
                     shared_memory: bool = True,
                     num_workers: int = None,
                     **kwargs) -> VectorEnv:
    """Create a vectorized environment from multiple copies of an environment.

    NOTE: This function does pretty much the same as `gym.vector.make`, but with
    a bit more flexibility:
    - Allows passing an env factory to start with, rather than only taking ids.
    - Allows passing wrappers to be added to the env on
        each worker, as well as wrappers to add on top of the returned (batched) env.
    - Allows passing tuples of (Type[Wrapper, ])
    - If `asynchronous` is `True` and the batch size is greater than the number
      of CPU cores on this machine, uses a `BatchedVectorEnv` with chunking 
      rather than a `AsyncVectorEnv`, to limit the the number of processes to
      the number of CPUs and increase performance.

    Parameters
    ----------
    base_env : str
        The environment ID (or an environment factory). This must be a valid ID
        from the registry.

    batch_size : int
        Number of copies of the environment (as well as batch size). 

    asynchronous : bool (default: `True`)
        If `True`, wraps the environments in an `AsyncVectorEnv` (which uses 
        `multiprocessing` to run the environments in parallel). If `False`,
        wraps the environments in a `SyncVectorEnv`.

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
    >>> env.reset()
    array([[-0.04456399,  0.04653909,  0.01326909, -0.02099827],
           [ 0.03073904,  0.00145001, -0.03088818, -0.03131252],
           [ 0.03468829,  0.01500225,  0.01230312,  0.01825218]],
          dtype=float32)
    """
    # Get the default wrappers, if needed.
    if isinstance(base_env, str) and not wrappers and use_default_wrappers_for_env:
        wrappers = default_wrappers_for_env.get(base_env, [])
    wrappers = wrappers or []
    
    base_env_factory: Callable[[], gym.Env]
    if isinstance(base_env, str):
        base_env_factory = partial(gym.make, base_env)
    elif callable(base_env):
        base_env_factory = base_env
    elif False and isinstance(base_env, gym.Env): # turning this off for now.
        # TODO: Check that there isn't a smarter way of doing this, maybe
        # by getting the env spec and creating a new one like it using
        # `gym.make`?
        logger.warning(RuntimeWarning(
            "Will try to use deepcopy as an env factory.. but this is really "
            "less than ideal!"))
        logger.debug(f"Env spec: {base_env.spec}")
        base_env_factory = partial(copy.deepcopy, base_env)
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

    if asynchronous:
        if num_workers is None:
            num_workers = mp.cpu_count()
        if batch_size != num_workers:
            return BatchedVectorEnv(env_fns, shared_memory=shared_memory, n_workers=num_workers)
        return AsyncVectorEnv(env_fns, shared_memory=shared_memory)
    return SyncVectorEnv(env_fns)


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
