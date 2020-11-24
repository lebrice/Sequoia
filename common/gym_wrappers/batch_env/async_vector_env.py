""" Tweaked version of the AsyncVectorEnv from openai gym that adds some methods
to be able to interact with the remote workers at a distance.
"""
import multiprocessing as mp
import operator
import platform
from enum import Enum
from functools import lru_cache, partial, wraps
from inspect import ismethod
from multiprocessing.connection import Connection
from operator import attrgetter, itemgetter, methodcaller
from typing import (Any, Callable, ClassVar, Dict, Generic, Iterable, List,
                    Optional, Sequence, Tuple, Type, TypeVar, Union, overload)

import gym
import numpy as np
from gym import Env, Wrapper
from gym.vector import AsyncVectorEnv as AsyncVectorEnv_
from gym.vector.async_vector_env import (AlreadyPendingCallError, AsyncState,
                                         NoAsyncCallError)
from utils.logging_utils import get_logger

from .tile_images import tile_images
from .worker import (CloudpickleWrapper, Commands, _custom_worker,
                     _custom_worker_shared_memory)
# NOTE: Seems to fix some kind of pytorch-related bug. I can try to find a link
# to the post about this if needed.
import os; os.environ['MKL_THREADING_LAYER'] = 'GNU'


logger = get_logger(__file__)
T = TypeVar("T")

class ExtendedAsyncState(Enum):
    WAITING_APPLY = "apply"


EnvType = TypeVar("EnvType", bound=gym.Env)

class AsyncVectorEnv(AsyncVectorEnv_, Sequence[EnvType]):
    
    # Whenever calling __getattr__ (when we're missing an attribute on the
    # VectorEnv), try to fetch the missing attribute from the remote workers.
    # If the attribute is a method, then this returns a BatchedMethod, which
    # when called with arguments, will execute the corresponding method on all
    # the remote environments. This could be particularly useful for doing
    # things like changing the task or seeding the remote workers, however it
    # adds some complexity, so I'm setting it to False by default. 
    allow_remote_getattr: ClassVar[bool] = False
    
    def __init__(self,
                 env_fns: Sequence[Callable[[], EnvType]],
                 context=None,
                 worker=None,
                 shared_memory=True,
                 **kwargs):
        if context is None:
            system: str = platform.system()
            if system == "Linux":
                context = "forkserver"
            else:
                logger.warning(RuntimeWarning(
                    f"Using the 'spawn' multiprocessing context since we're on "
                    f"a non-linux {system} system. This means creating new "
                    f"worker processes will probably be quite a bit slower. "
                ))
                context = "spawn"

        # TODO: @lebrice If we want to be able to add back the cool things we
        # had before, like remotely modifying the envs' attributes, only
        # resetting a portion of them, etc, we'll have to take a look at the
        # worker_ function, copy it into `worker.py`, modify it, and then change
        # the value of `worker` here.
        if worker is None:
            if shared_memory:
                worker = _custom_worker_shared_memory
            else:
                worker = _custom_worker
        # List that stores wether a function is being applied on an env and we
        # should expect a result response for that env.
        self.expects_result: List[bool] = []
        
                
        # Important, this must be done before the call to super().__init__
        from common.spaces.sparse import Sparse
        
        super().__init__(
            env_fns=env_fns,
            context=context,
            worker=worker,
            shared_memory=shared_memory,
            **kwargs
        )

    def random_actions(self) -> Tuple:
        return self.action_space.sample()

    def __len__(self) -> int:
        return self.num_envs

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        self._assert_is_running()
        for pipe in self.parent_pipes:
            pipe.send(('render', None))
        
        image_batch = np.stack([pipe.recv() for pipe in self.parent_pipes])
        if mode == "rgb_array":
            return image_batch
        
        if mode == "human":
            tiled_version = tile_images(image_batch)
            if self.viewer is None:
                from gym.envs.classic_control import rendering
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(tiled_version)
            return self.viewer.isopen
        
        raise NotImplementedError(f"Unsupported mode {mode}")
        
        
        # NOTE: @lebrice This used to be working, and would have been an example
        # use-case for all the fancy stuff written below, which I should
        # probably remove at some point if we don't end up needing access to the
        # envs. 
        return np.asarray(self[:].render(mode="rgb_array"))

    @overload
    def apply(self, functions: Callable[[Env], T]) -> List[T]:
        ...

    @overload
    def apply(self, functions: Sequence[Callable[[Env], T]]) -> List[T]:
        ...

    @overload
    def apply(self, functions: Sequence[Optional[Callable[[Env], T]]]) -> List[Optional[T]]:
        ...

    def apply(self,
              functions: Union[Callable[[Env], T], Sequence[Optional[Callable[[Env], T]]]],
              timeout: float = None) -> List[T]:
        """ Send a function down to the workers for them to apply to their
        environments, and returns the corresponding results.

        If given a single function, applies the same function to all the envs.
        When given a list of functions, apples each function to each env.
        When given a list where some items aren't callables, e.g. None, doesn't
        apply any function for that particular env.
        """
        self.apply_async(functions)
        return self.apply_wait(timeout=timeout)

    def apply_async(self, functions: Union[Callable[[Env], Any], Sequence[Callable[[Env], Any]]]):
        self._assert_is_running()
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError('Calling `apply` while waiting '
                'for a pending call to `{0}` to complete.'.format(
                self._state.value), self._state.value)

        if callable(functions):
            functions = [functions] * self.num_envs
        assert len(functions) == self.num_envs, "Need a function for each env."

        self.expects_result.clear()
        for pipe, function in zip(self.parent_pipes, functions):
            if callable(function):
                self.expects_result.append(True)
                pipe.send((Commands.apply, function))
            else:
                self.expects_result.append(False)
        self._state = ExtendedAsyncState.WAITING_APPLY
        
    def apply_wait(self, timeout: float = None) -> List[Optional[Any]]:
        # Could split this into an 'apply_async' and 'apply_wait' if we wanted
        # to. setting the self._state attribute isn't really needed here.
        self._assert_is_running()
        if self._state != ExtendedAsyncState.WAITING_APPLY:
            raise NoAsyncCallError('Calling `apply_wait` without any prior call '
                'to `step_async`.', ExtendedAsyncState.WAITING_APPLY.value)

        if not self._poll(timeout):
            self._state = AsyncState.DEFAULT
            raise mp.TimeoutError('The call to `apply_wait` has timed out after '
                '{0} second{1}.'.format(timeout, 's' if timeout > 1 else ''))
        
        results: List[Any] = []
        successes: List[bool] = []
        pipe: Connection
        for pipe, need_result in zip(self.parent_pipes, self.expects_result):
            if need_result:
                result, success = pipe.recv()
            else:
                result, success = None, True
            results.append(result)
            successes.append(success)

        self._raise_if_errors(successes)
        self._state = AsyncState.DEFAULT
        return list(results)

    @overload
    def apply_at(self, operation: Callable[[EnvType], T], index: int) -> T:
        ...
    
    @overload
    def apply_at(self, operation: Callable[[EnvType], T], index: Sequence[int]) -> List[T]:
        ...

    def apply_at(self,
                 operation: Callable[[EnvType], T],
                 index: Union[int, Sequence[int]]) -> Union[T, List[T]]:
        """ Applies `operation` to the envs at `index`. """
        indices = [index] if isinstance(index, int) else index
        operations = [
            operation if i in indices else None for i in range(self.num_envs)
        ]
        results: List[Optional[T]] = self.apply(operations)
        assert len(results) == self.num_envs

        if isinstance(index, int):
            # If we wanted a proxy for a single item, then we return a
            # single result, instead of a list with one item.
            return results[index]
        return [result for i, result in enumerate(results) if i in indices]


    def __getattr__(self, name: str):
        if name in {"closed", "_state"}:
            return

        if not type(self).allow_remote_getattr:
            raise AttributeError(name)
        
        logger.debug(f"Attempting to get missing attribute {name}.")
        
        assert isinstance(name, str)
        env_has_attribute = self.apply(partial(hasattr_, name=name))
        if all(env_has_attribute):
            self._assert_is_running()
            return getattr(self[:], name)
        raise AttributeError(name)

    def __getitem__(self, index: Union[int, slice, Sequence[int]]) -> EnvType:
        if isinstance(index, slice):
            index = tuple(range(self.num_envs))[index]
        elif isinstance(index, list):
            index = tuple(index)
        elif isinstance(index, np.ndarray):
            if index.dtype == np.bool:
                index = np.arange(self.num_envs)[index]
            index = tuple(index.tolist())
        elif not isinstance(index, int):
            try:
                index = tuple(index)
            except:
                raise RuntimeError(f"Bad index: {index}")
        return self.__get_env_proxy(index)

    @lru_cache()
    def __get_env_proxy(self, index: Union[int, Tuple[int, ...]]) -> EnvType:
        """ Returns a Proxy object that will get/set attributes on the remote
        environments at the given indices.
        """
        apply_at_indices = partial(self.apply_at, index=index)
        from .batched_method import BatchedMethod

        class Proxy:
            """ Some Pretty sweet functional magic going on here.

            NOTE: @lebrice: Since I don't want (or need) a 'self' argument in
            the methods below, I marked all the methods as static.
            TODO: Maybe be useful to read-up on the descriptor protocol.
            """
            @staticmethod
            def __getattribute__(name: str) -> List:
                """ Gets the attribute from the corresponding remote env, rather
                than from this proxy object.
                """
                results = apply_at_indices(attrgetter(name))
                if isinstance(results, list) and all(map(ismethod, results)):
                    # Detect when the requested attributes are methods, and then
                    # batch the methods!
                    return BatchedMethod(results, apply_methods_fn=apply_at_indices)
                return results

            @staticmethod
            def __setattr__(name: str, value: Any):
                """ Sets the attribute on the corresponding remote env, rather
                than on this proxy object.
                """
                # TODO: IF the value is a list, and index is a tuple of more
                # than one value, then maybe split the value up to set a
                # different slice of it on each env ?
                return apply_at_indices(partial(set_wrapper_attribute, name=name, value=value))

            @staticmethod
            def getattributes(*name: str) -> List:
                """ Bulk getattr to save some latency. """
                return apply_at_indices(attrgetter(*name))

            @staticmethod
            def setattributes(**names_and_values):
                """ Bulk setattr to save some latency. """
                return apply_at_indices(partial(setattrs, **names_and_values))

            @staticmethod
            def __getitem__(index: int):
                return apply_at_indices(itemgetter(index))
            # Pretty sure this wouldn't be used, but just trying to see if
            # there's a pattern here we can make use of, hopefully involving the
            # use of `methodcaller` from the `operator` package!
            @staticmethod
            def __add__(val):
                return apply_at_indices(partial(operator.add, val))

        # Return such a proxy object.
        return Proxy()

        raise NotImplementedError(
            "TODO: Return an object that, when set an attribute or getting an "
            "attribute on it, will actually instead asks the corresponding env "
            "at that index for that attribute, or set the attribute on that "
            " env, something like that."
        )

def hasattr_(obj, name) -> None:
    """ Version of 'hasattr' that accepts keyword arguments, for use with partial.
    """
    assert isinstance(name, str) 
    return hasattr(obj, name)


def setattr_(obj, name, value) -> None:
    """ Version of 'setattr' that accepts keyword arguments, for use with partial.
    """
    setattr(obj, name, value)


def setattrs(obj, **name_and_value) -> None:
    """ Version of 'setattr' that accepts keyword arguments, for use with partial.
    """
    for name, value in name_and_value.items():
        setattr(obj, name, value)



def setattr_on_unwrapped(env: gym.Env, attr: str, value: Any) -> None:
    setattr(env.unwrapped, attr, value)


def setattrs_on_unwrapped(env: gym.Env, **names_and_values) -> None:
    for name, value in names_and_values.items():
        setattr_on_unwrapped(env, name, value)

def set_wrapper_attribute(env: Env, name: str, value: Any) -> Type[Env]:
    """ Sets the attribute `name` to a value of `value` on the first wrapper
    that already has it.
    If none have it, sets the attribute on the unwrapped env.

    Returns the type of the object on which the attribute was set.
    TODO: not sure if this return value is really useful. I added it just to be
    able to tell if it was set on the right wrapper, in case more than one
    wrapper has an attribute with that name.
    """
    # Keep track of seen envs to avoids infinite loops because of cycles.
    wrappers = []
    while not hasattr(env, name) and hasattr(env, "env") and env not in wrappers:
        wrappers.append(env)
        env = env.env
    setattr(env, name, value)
    return type(env)


def set_wrapper_attributes(env: Env, **names_and_values) -> Dict[str, Type[Env]]:
    results: Dict[str, Type[Wrapper]] = {}
    for name, value in names_and_values.items():
        results[name] = set_wrapper_attribute(env, name, value)
    return results
