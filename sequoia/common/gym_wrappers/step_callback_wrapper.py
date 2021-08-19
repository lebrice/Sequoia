"""TODO: Make a wrapper that calls a given function/callback when a given step is reached.
"""
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import wraps
from typing import Callable, Dict, List, Tuple, Union
from .utils import IterableWrapper
import gym


class StepCallbackWrapper(IterableWrapper):
    """ Wrapper that will execute some callbacks when certain steps are reached.
    """
    def __init__(self,
                 env: gym.Env,
                 callbacks: List[Callable] = None,
                 ):
        super().__init__(env, call_hooks=True)
        self._steps = 0
        self.callbacks = callbacks or []

    def add_callback(self, callback: Callable) -> None:
        """ Adds a callback that will be executed at each step. """
        self.callbacks.append(callback)

    def add_step_callback(self, step: int, callback: Callable):
        @wraps(callback)
        def _func(_step: int,*args, **kwargs):
            if _step == step:
                # print(f"Running callback {callback} at step {_step}")
                callback(_step, *args, **kwargs)
        self.add_callback(_func)

    def add_periodic_callback(self, callback: Callable, period: int, offset: int = 0):
        @wraps(callback)
        def _periodic_func(_step: int, *args, **kwargs):
            if _step % period == offset:
                callback(_step, *args, **kwargs)
        self.add_callback(_periodic_func)

    def action(self, action):
        action = super().action(action)
        self._callback_loop()
        self._steps += 1
        return action

    def observation(self, observation):
        return super().observation(observation)

    def step(self, action):
        # self._callback_loop(observation=observation)
        return super().step(action)

    def _callback_loop(self, *args, **kwargs):
        for callback in self.callbacks:
            # if it's a callable, just call it all the time, assuming that
            # it will use some condition in it's __call__ to check wether
            # it should be executed or not.
            callback(self._steps, *args, **kwargs)
