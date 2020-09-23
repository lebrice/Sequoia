
import math
import multiprocessing as mp
import platform
from functools import partial
from inspect import ismethod
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection, wait
from typing import (Any, Callable, Iterable, List, Optional, Sequence, Tuple,
                    TypeVar, Union)

import gym
import numpy as np
import torch
from gym.vector import AsyncVectorEnv as AsyncVectorEnv_
from torch import Tensor
from torch.utils.data import IterableDataset

from utils import n_consecutive
from utils.logging_utils import get_logger

from .async_vector_env import AsyncVectorEnv
from .worker import CloudpickleWrapper, Commands, custom_worker

_missing = object()
T = TypeVar("T")

class BatchEnv(AsyncVectorEnv):
    # TODO: This is what I was using in the previous implementation of the BatchEnv.
    def getattr(self, attr: str) -> List:
        """Gets the value of the given attribute from all environments.  

        Args:
            attr (str): The attribute to fetch.

        Returns:
            List: The list of values for each environment.
        """
        return self.get_attr_from_envs(attr)

    def setattr(self, attr: str, value: Any) -> None:
        """Sets the given attribute to the given value on all the environments. 
                
        NOTE: It's important to use this method rather than just setting the
        attribute as usual, as this would instead set the value on the `BatchEnv`
        object rather than on the remotes! For example, writing `env.i = 123`
        doesn't actually set the `i` attribute on all the envs, it just creates
        a new `i` attribute on the `BatchEnv` object.
        TODO: Maybe we could use something like the __slots__ magic to fix this?

        Args:
            attr (str): Name of the attribute to set.
            value (Any): Value to be set on all environments.
        """
        self.set_attr_on_envs(attr, value)

    def setattr_foreach(self, attr: str, values: Sequence) -> None:
        """ Sets `attr` on each env to the corresponding value from `values`. 

        Roughly equivalent to the following pseudocode (minus the mp stuff):
        ```
        for env, value in zip(self.envs, values):
            setattr(env, attr, value)
        ```

        Args:
            attr (str): Attribute to be set.
            values (Sequence): Values for each environment. Must have the same
                length as the number of environments (`self.batch_size`), else
                an error is raised.
        """
        self.set_attr_on_each_env(attr, values)



    def __getattr__(self, attr: str, default: Any=_missing) -> Union[Any, List[Any], Callable]:
        logger.debug(f"Trying to get missing attribute '{attr}'.")
        # TODO: This is causing problems atm.
        attributes: List = []
        if default is _missing:
            attributes = self.get_attr_from_envs(attr)
        else:
            try:
                attributes = self.get_attr_from_envs(attr)
            except AttributeError:
                return default

        logger.debug(f"Attributes: {attributes}")
        # TODO: Having some fun here, should turn keep this off just in case
        # there's any problem. 
        if False and all(map(ismethod, attributes)):
            logger.warning(RuntimeWarning(
                f"The '{attr}' attribute is a method on all envs, returning a "
                "'batched' method, just for fun's sake."
            ))
            from .batched_method import make_batched_method
            return make_batched_method(attributes)
        return attributes

    def get_attr_from_envs(self, attr: str) -> List[Any]:
        for remote in self.remotes:
            remote.send((Commands.get_attr, attr))
        results = []
        for i, remote in enumerate(self.remotes):
            remote_results: Sequence = remote.recv().x
            logger.debug(f"Results from remote #{i}: {remote_results}")
            for result in remote_results:
                if isinstance(result, AttributeError):
                    raise result
                results.append(result)
        logger.debug(f"Results: {results}")
        return results

    def set_attr_on_envs(self, attr: str, value: Any) -> None:
        logger.debug(f"Will try to set attribute '{attr}' to a value of {value} in all environments.")
        for remote in self.remotes:
            remote.send((Commands.set_attr, CloudpickleWrapper((attr, value))))
        for remote in self.remotes:
            # We expect to return a 'None', even though it isn't strictly
            # necessary, just so we know everything went well.
            remote_results: List = remote.recv()
            for result in remote_results:
                if result is not None:
                    raise RuntimeError(
                        f"Something went wrong when trying to set attribute "
                        f"{attr}: {result}"
                    )

    def set_attr_on_each_env(self, attr: str, values: Sequence) -> None:
        logger.debug(f"Will try to set attribute '{attr}' with the corresponding values for each env.")
        # Just in case we're given a generator or iterable.
        values = list(values)
        if len(values) != self.num_envs:
            raise RuntimeError(
                f"You need to pass a value for each of the {self.num_envs} "
                f"environments. (received {values})"
            )

        # Make a list of the values for each remote.
        values_per_remote: List[Tuple] = list(n_consecutive(values, self.in_series))

        for remote, values_for_remote in zip(self.remotes, values_per_remote):
            args = CloudpickleWrapper((attr, values_for_remote))
            remote.send((Commands.set_attr_on_each, args))

        for remote in self.remotes:
            # We expect to return a 'None', even though it isn't strictly
            # necessary, just so we know everything went well.
            remote_results: List = remote.recv()
            for result in remote_results:
                if result is not None:
                    raise RuntimeError(f"Something went wrong when trying to set attribute {attr}: {result}")

    def partial_reset(self, reset_mask: Sequence[bool]) -> List[Optional[Any]]:
        values_per_remote: List[Tuple[bool, ...]] = self.split_values(reset_mask)
        # Make a list of the values for each remote.
        self._assert_not_closed()
        for remote, values_for_remote in zip(self.remotes, values_per_remote):
            args = CloudpickleWrapper(values_for_remote)
            remote.send((Commands.partial_reset, args))

        results = []
        for remote in self.remotes:
            # We expect to receive None for the envs that weren't reset, and the
            # reset state for those that were.
            remote_results: List = remote.recv()
            if isinstance(remote_results, CloudpickleWrapper):
                remote_results = remote_results.x
            results.extend(remote_results)
        return zip(*results)

    def split_values(self, values: List[T]) -> List[Tuple[T, ...]]:
        # Make a list of the values for each remote.
        values = list(values) # in case it's a generator or something.
        if len(values) != self.num_envs:
            raise RuntimeError(
                f"You need to pass a value for each of the {self.num_envs} "
                f"environments, only received {len(values)} values."
            )
        values_per_remote = list(n_consecutive(values, self.in_series))
        return values_per_remote

    def seed(self, seeds: Union[int, Iterable[int]]) -> None:
        if isinstance(seeds, int):
            seeds = [seeds] * self.num_envs
        seeds = self.split_values(seeds)
        for remote, seeds_for_remote in zip(self.remotes, seeds):
            remote.send((Commands.seed, seeds_for_remote))
        for remote in self.remotes:
            remote.recv()
