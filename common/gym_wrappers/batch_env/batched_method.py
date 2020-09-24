""" An absolute orgy of functional programming goodness.

Author: @lebrice

This could be used to automatically batch the method calls on objects that are 'split'.
for instace, doing something like
`env.seed([123] * batch_size)` to seed multiple sub-environments.

Another crazy cool idea is, if we had something like 
`env.action_space.sample()`, with `env` not even having an action space itself,
we could get a "batched method" for the `action_space` part, and then calling
`sample` would then return a list of samples, one per action space.
"""
from inspect import ismethod
from typing import Callable, List

from utils.logging_utils import get_logger
from operator import methodcaller
logger = get_logger(__file__)


class BatchedMethod(List[Callable]):
    def __init__(self, *args, apply_methods_fn: Callable[[List[Callable]], List] = None):
        super().__init__(*args)
        # self.method_name: str = method_name
        self.apply_methods_fn = apply_methods_fn
    
    @property
    def method_name(self) -> str:
        return self[0].__name__ if len(self) > 1 else None

    def __call__(self, *args, split_args: bool = False, **kwargs):
        print(f"args: {args}, kwargs: {kwargs}")
        if not self.method_name:
            return []
        if self.apply_methods_fn:
            results = self.apply_methods_fn(methodcaller(self.method_name, *args, **kwargs))
        else:
            results = [method(*args, **kwargs) for method in self]
        logger.debug(f"Results: {results}")
        if isinstance(results, list) and all(map(ismethod, results)):
            logger.debug("Recurse?! :DD")
            # TODO: Need to get the method name from `self`..
            return BatchedMethod(results, apply_methods_fn=self.apply_methods_fn)
        return results

def make_batched_method(methods: List[Callable]) -> Callable:
    def batched_method(*args, **kwargs):
        batched_args = tuple(zip(*args))
        batched_kwargs = {
            k: tuple(zip(*v)) for k, v in kwargs.items()
        }
        logger.debug(f"batched args: {batched_args}, kwargs: {batched_kwargs}")
        results = []
        for i, method in enumerate(methods):
            args_i = batched_args[i]
            kwargs_i = {k: v[i] for k, v in batched_kwargs.items()}
            result = method(*args_i, **kwargs_i)
            results.append(result)
        if all(map(ismethod, results)):
            logger.debug("Recurse?! :DD")
            return make_batched_method(results)
        return results
    return batched_method
