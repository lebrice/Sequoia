import multiprocessing as mp
import operator
import platform
from enum import Enum
from functools import lru_cache, partial, wraps
from multiprocessing.connection import Connection
from operator import attrgetter, itemgetter, methodcaller
from typing import (Any, Callable, Generic, Iterable, List, Optional, Sequence,
                    Tuple, TypeVar, Union, overload)

import gym
from gym import Env
from gym.vector import AsyncVectorEnv as AsyncVectorEnv_
from gym.vector.async_vector_env import (AlreadyPendingCallError, AsyncState,
                                         NoAsyncCallError)

from utils.logging_utils import get_logger

from .worker import (CloudpickleWrapper, Commands,
                     _custom_worker_shared_memory, custom_worker)

logger = get_logger(__file__)
T = TypeVar("T")

class ExtendedAsyncState(Enum):
    WAITING_APPLY = "apply"


EnvType = TypeVar("EnvType", bound=gym.Env)

class AsyncVectorEnv(AsyncVectorEnv_, Sequence[EnvType]):
    
    def __init__(self,
                 env_fns: Sequence[Callable[[], EnvType]],
                 context=None,
                 worker=None,
                 **kwargs):
        if context is None:
            system: str = platform.system()
            if system == "Linux":
                # TODO: Debugging an error from the pyglet package when using 'fork'.
                # python3.7/site-packages/pyglet/gl/xlib.py", line 218, in __init__
                # raise gl.ContextException('Could not create GL context')
                # context = "fork"
                # context = "spawn"
                # NOTE: For now 'forkserver`, seems to have resolved the bug
                # above for now:
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
            worker = _custom_worker_shared_memory

        self.expects_result: List[bool] = []
        super().__init__(
            env_fns=env_fns,
            context=context,
            worker=worker,
            **kwargs
        )

    def random_actions(self) -> Tuple:
        return self.action_space.sample()

    def __len__(self) -> int:
        return self.num_envs

    @overload
    def apply(self, functions: Callable[[Env], T]) -> List[T]:
        ...

    @overload
    def apply(self, functions: Sequence[Callable[[Env], T]]) -> List[T]:
        ...
    
    @overload
    def apply(self, functions: Sequence[Optional[Callable[[Env], T]]]) -> List[Optional[T]]:
        ...

    def apply(self, functions: Union[Callable[[Env], T], Sequence[Optional[Callable[[Env], T]]]]) -> List[T]:
        """ Send a function down to the workers for them to apply to their
        environments, and returns the corresponding results.

        If given a single function, applies the same function to all the envs.
        When given a list of functions, apples each function to each env.
        When given a list where some items aren't callables, e.g. None, doesn't
        apply any function for that particular env.
        """
        self.apply_async(functions)
        return self.apply_wait()

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
        self.step_wait
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

    def __getattr__(self, name: str):
        env_has_attribute = self.apply(partial(has_attribute, name=name))
        if all(env_has_attribute):
            return getattr(self[:], name)


    def __getitem__(self, index: Union[int, slice, Sequence[int]]) -> EnvType:
        if isinstance(index, int):
            pass
        elif isinstance(index, slice):
            index = tuple(range(self.num_envs))[index]
        elif isinstance(index, list):
            index = tuple(index)
        else:
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
        indices: List[int] = []
        if isinstance(index, int):
            indices = [index]
        else:
            try:
                indices = list(index)
            except:
                raise RuntimeError(f"Bad index: {index}")

        def apply_at_indices(operation: Callable) -> List:
            """ Version of 'apply' but only for only the indices in `indices`.
            """
            operations = [
                operation if i in indices else None for i in range(self.num_envs)
            ]
            results = self.apply(operations)
            assert isinstance(results, list) and len(results) == self.num_envs
            if isinstance(index, int):
                # If we wanted a proxy for a single item, then we return a
                # single result, not a list with one item.
                return results[index]
            return [result for i, result in enumerate(results) if i in indices] 

        class Proxy:
            """ Some Pretty sweet functional magic going on here.

            NOTE: @lebrice: Since I don't want (or need) a 'self' argument in
            the methods below, I marked all the methods as static.
            TODO: Maybe I should read-up on the descriptor protocol, sounds
            relevant.
            """
            @staticmethod
            def __getattr__(name: str) -> List:
                """ Gets the attribute from the corresponding remote env, rather
                than from this proxy object.
                """
                # If we wanted to be even weirer about this, we could try and
                # detect whenever such an attribute would be a method, and then
                # batch the methods!
                return apply_at_indices(attrgetter(name))

            @staticmethod
            def __setattr__(name: str, value: Any):
                """ Sets the attribute on the corresponding remote env, rather
                than on this proxy object.
                """
                # TODO: IF the value is a list, and index is a tuple of more
                # than one value, then maybe split the value up to set a
                # different slice of it on each env ?
                return apply_at_indices(partial(set_attribute, name=name, value=value))

            @staticmethod
            def getattributes(*name: str) -> List:
                """ Bulk getattr to save some latency. """
                return apply_at_indices(attrgetter(*name))

            @staticmethod
            def setattributes(**names_and_values):
                """ Bulk setattr to save some latency. """
                return apply_at_indices(partial(set_attributes, **names_and_values))

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

def set_attribute(obj, name, value) -> None:
    """ Version of 'setattr' that accepts keyword arguments, for use with partial.
    """
    setattr(obj, name, value)


def has_attribute(obj, name) -> None:
    """ Version of 'hasattr' that accepts keyword arguments, for use with partial.
    """
    return hasattr(obj, name)


def set_attributes(obj, **names_and_values) -> None:
    for name, value in names_and_values.items():
        setattr(obj, name, value)

def attrsetter(name, val):
    # def setter(obj):
    #     setattr(obj, name, val)
    return lambda obj: setattr(obj, name, val)
