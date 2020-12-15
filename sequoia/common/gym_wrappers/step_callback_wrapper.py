"""TODO: Make a wrapper that calls a given function/callback when a given step is reached.
"""
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from typing import Callable, Dict, List, Tuple, Union
from .utils import IterableWrapper
import gym


class Callback(Callable[[int, gym.Env], None], ABC):
    @abstractmethod
    def __call__(self, step: int, env: gym.Env, step_results: Tuple) -> None:
        raise NotImplementedError()


class StepCallback(Callback, ABC):
    def __init__(self, step: int, func: Callable[[int, gym.Env, Tuple], None] = None):
        self.step = step
        self.func = func

    def __call__(self, step: int, env: gym.Env, step_results: Tuple) -> None:
        if self.func:
            return self.func(step, env, step_results)
        raise NotImplementedError("Create your own callback or pass a func to use.")

class PeriodicCallback(Callback):
    def __init__(self,
                 period: int,
                 offset: int = 0,
                 func: Callable[[int, gym.Env], None] = None):
        self.period = period
        self.offset = offset
        self.func = func
    
    def __call__(self, step: int, env: gym.Env, step_results: Tuple) -> None:
        if self.func:
            return self.func(step, env, step_results)
        raise NotImplementedError("Create your own callback or pass a func to use.")


class StepCallbackWrapper(IterableWrapper):
    """ Wrapper that will execute some callbacks when certain steps are reached.
    """
    def __init__(self,
                 env: gym.Env,
                 callbacks: List[Callback] = None,
                 ):
        super().__init__(env)
        self._steps = 0
        self.callbacks = callbacks or []

    def add_callback(self, callback: Union[Callback]) -> None:
        self.callbacks.append(callback)

    def add_step_callback(self, step: int, callback: Callable[[int, gym.Env], None]):
        if isinstance(callback, StepCallback):
            assert step == callback.step
        else:
            callback = StepCallback(step=step, func=callback)
        self.add_callback(callback)

    def add_periodic_callback(self, period: int, callback: StepCallback, offset: int = 0):
        if isinstance(callback, PeriodicCallback):
            assert period == callback.period
            assert offset == callback.offset
        else:
            callback = PeriodicCallback(period=period, offset=offset, func=callback)
        self.add_callback(callback)

    def step(self, action):
        step_results = super().step(action)
        for callback in self.callbacks:
            if isinstance(callback, StepCallback):
                if callback.step == self._steps:
                    callback(self._steps, self, step_results)
            elif isinstance(callback, PeriodicCallback):
                if (self._steps >= callback.offset and
                       (self._steps - callback.offset) % callback.period == 0):
                    callback(self._steps, self, step_results)
            else:
                # if it's a callable, just call it all the time, assuming that
                # it will use some condition in it's __call__ to check wether
                # it should be executed or not.
                callback(self._steps, self, step_results)
        self._steps += 1
        return step_results
