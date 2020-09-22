import math
import multiprocessing as mp
from functools import partial
from inspect import ismethod
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection, wait
from typing import (Any, Callable, Iterable, List, Sequence, Tuple, TypeVar,
                    Union)

import gym
import numpy as np
import torch
from torch.utils.data import IterableDataset

from utils.logging_utils import get_logger

from .subproc_vec_env import _SubprocVecEnv
from .worker import custom_worker

logger = get_logger(__file__)
T = TypeVar("T")
_missing = object()

class BatchEnv(gym.Wrapper, IterableDataset):
    __slots__: List[str] = ["env", "batch_size"]
    def __init__(self,
                 env: str = None,
                 env_factory: Callable[[int], gym.Env] = None,
                 batch_size: int = 1,
                 num_envs_per_worker: int = 1,
                 auto_reset: bool = False,
                 ):
        """Creates a batched environment using multiprocessing workers.

        Args:
            env (str, optional): gym env identifier. Defaults to None.
            env_factory (Callable[[int], gym.Env], optional): Factory function
                that takes no parameters and returns a gym
                environment. Defaults to None.
            batch_size (int, optional): Batch size (number of environments).
                Defaults to None.
            num_envs_per_worker (int, optional): Number of environments per
                worker. Defaults to 1.
        """

        assert (isinstance(env, str) or env_factory), (
            "Must pass either a string env or an env_factory callable."
        )
        assert batch_size >= 1, f"batch_size must be greater or equal to 1 (got {batch_size})."
        if not env_factory:
            env_factory = partial(gym.make, env)
        env_fns = [env_factory for i in range(batch_size)]
        env = _SubprocVecEnv(
            env_fns=env_fns,
            worker=partial(custom_worker, auto_reset=auto_reset),
            in_series=num_envs_per_worker,            
        )
        super().__init__(env)
        self.env: _SubprocVecEnv
        self.batch_size = batch_size
    
    def random_actions(self) -> Sequence:
        return np.stack([
            self.env.observation_space.sample() for _ in range(self.batch_size)
        ])

    def getattr(self, attr: str) -> List:
        """Gets the value of the given attribute from all environments.  

        Args:
            attr (str): The attribute to fetch.

        Returns:
            List: The list of values for each environment.
        """
        return self.env.get_attr_from_envs(attr)

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
        self.env.set_attr_on_envs(attr, value)

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
        self.env.set_attr_on_each_env(attr, values)
