""" Creates an IterableDataset from a Gym Environment. 
"""
from collections import abc as collections_abc
from typing import (Any, Callable, Dict, Generator, Generic, Iterable, List,
                    Optional, Sequence, Tuple, Type, TypeVar, Union)
import warnings
import gym
import numpy as np
from torch import Tensor
from torch.utils.data import IterableDataset

from utils.logging_utils import get_logger

from .batch_env import VectorEnv
from .utils import ActionType, ObservationType, RewardType, StepResult
# from settings.base.objects import Observations, Rewards, Actions
logger = get_logger(__file__)


Item = TypeVar("Item")


class EnvDataset(gym.Wrapper,
                 IterableDataset,
                 Generic[ObservationType, ActionType, RewardType, Item],
                 Iterable[Item]):
    """ Wrapper that exposes a Gym environment as an IterableDataset.

    This makes it possible to iterate over a gym env with an Active DataLoader.
    
    One pass through __iter__ is one episode. The __iter__ method can be called
    at most `max_episodes` times.
    """
    def __init__(self,
                 env: gym.Env,
                 max_steps: Optional[int] = None,
                 max_episodes: Optional[int] = None,
                 max_steps_per_episode: Optional[int] = None,
                 ):
        super().__init__(env=env)
        if isinstance(env.unwrapped, VectorEnv):
            if not max_steps_per_episode:
                warnings.warn(UserWarning(
                    "Iterations through the dataset (episodes) could be "
                    "infinitely long, since the env is a VectorEnv and "
                    "max_steps_per_episode wasn't given!"
                ))

        # Maximum number of episodes per iteration.
        self.max_episodes = max_episodes
        # Maximum number of steps per iteration.
        self.max_steps = max_steps
        self.max_steps_per_episode = max_steps_per_episode
        
        # Number of steps performed in the current episode.
        self._n_steps_in_episode: int = 0
        
        # Total number of steps performed so far.
        self._n_steps: int = 0
        # Number of episodes performed in the environment.
        # Starts at -1 so the initial _reset doesn't count as the end of an episode.
        self._n_episodes: int =  0
        # Number of times the `send` method was called.
        self._n_sends: int = 0
        
        self._observation: Optional[ObservationType] = None
        self._action: Optional[ActionType] = None
        self._reward: Optional[RewardType] = None
        self._done: Optional[Union[bool, Sequence[bool]]] = None
        self._info: Optional[Union[Dict, Sequence[Dict]]] = None

        self._closed: bool = False
        self._reset: bool = False
        
        self._current_step_result: StepResult = None
        self._previous_step_result: StepResult = None

    def reset_counters(self):
        self._n_steps = 0
        self._n_episodes = 0
        self._n_sends = 0
        self._n_steps_in_episode = 0

    def step(self, action) -> StepResult:
        if self._closed:
            if self.reached_episode_limit:
                raise gym.error.ClosedEnvironmentError(f"Env has already reached episode limit ({self.max_episodes}) and is closed.")
            elif self.reached_step_limit:
                raise gym.error.ClosedEnvironmentError(f"Env has already reached step limit ({self.max_steps}) and is closed.")
            else:
                raise gym.error.ClosedEnvironmentError("Can't call step on closed env.")
        result = StepResult(*self.env.step(action))
        self._n_steps += 1
        self._n_steps_in_episode += 1
        return result

    def __next__(self) -> Tuple[ObservationType,
                                Union[bool, Sequence[bool]],
                                Union[Dict, Sequence[Dict]]]:
        """Produces the next observations, or raises StopIteration.

        Returns
        -------
        Tuple[ObservationType, Union[bool, Sequence[bool]], Union[Dict, Sequence[Dict]]]
            [description]

        Raises
        ------
        gym.error.ClosedEnvironmentError
            If the env is already closed.
        gym.error.ResetNeeded
            If the env hasn't been reset before this is called.
        StopIteration
            When the step limit has been reached.
        StopIteration
            When the episode limit has been reached.
        RuntimeError
            When an action wasn't passed through 'send', and a default policy
            isn't set.
        """
        logger.debug(f"__next__ is being called at step {self._n_steps}.")
        
        if self._closed:
            raise gym.error.ClosedEnvironmentError("Env is closed.")
        
        if self.reached_episode_limit:
            logger.debug("Reached episode limit, raising StopIteration.")
            raise StopIteration
        if self.reached_step_limit:
            logger.debug("Reached step limit, raising StopIteration.")
            raise StopIteration
        if self.reached_episode_length_limit:
            logger.debug("Reached episode length limit, raising StopIteration.")

        if not self._reset:
            raise gym.error.ResetNeeded("Need to reset the env before you can call __next__")
        
        if self._action is None:
            raise RuntimeError(
                "You have to send an action using send() between every "
                "observation."
            )
        self._observation, self._reward, self._done, self._info = self.step(self._action)
        return self._observation

    def send(self, action: ActionType) -> RewardType:
        """ Sends an action to the environment, returning a reward.
        This will raise an error when if not called without
        """
        assert action is not None, "Don't send a None action!"
        self._action = action
        self._observation = self.__next__()
        self._n_sends += 1
        return self._reward

    def __iter__(self) -> Iterable[Tuple[ObservationType,
                                         Union[bool, Sequence[bool]],
                                         Union[Dict, Sequence[Dict]]]]:
        """Iterator for an episode of a gym environment.
        """
        
        if self._closed:
            if self.reached_episode_limit:
                raise gym.error.ClosedEnvironmentError(f"Env has already reached episode limit ({self.max_episodes}) and is closed.")
            elif self.reached_step_limit:
                raise gym.error.ClosedEnvironmentError(f"Env has already reached step limit ({self.max_steps}) and is closed.")
            else:
                raise gym.error.ClosedEnvironmentError(f"Env is closed, can't iterate over it.")
        return self.iterator_with_send()

    def iterator_with_send(self) -> Iterable[ObservationType]:
        """Iterator for an episode in the environment, which uses the 'active
        dataset' style with __iter__ and send.

        Yields
        -------
        Observations
            Observations from the environment.

        Raises
        ------
        RuntimeError
            [description]
        """
        # First step:
        if not self._reset:
            self._observation = self.reset()
        self._done = False
        self._action = None
        self._reward = None
        assert self._observation is not None
        # Yield the first observation.
        # TODO: What do we want to yield, actually? Just observations?
        yield self._observation

        logger.debug(f"episode {self._n_episodes}/{self.max_episodes}")

        while not any([self._done_is_true,
                       self.reached_step_limit,
                       self.reached_episode_length_limit]):
            logger.debug(f"step {self._n_steps}/{self.max_steps}, ")
            
            # Set those to None to force the user to call .send()
            self._action = None
            self._reward = None
            yield self._observation

            if self._action is None:
                raise RuntimeError(
                    "You have to send an action using send() between every "
                    "observation."
                )

                
        # Force the user to call reset() between episodes.
        self._reset = False
        self._n_episodes += 1

        logger.debug(f"self.n_steps: {self._n_steps} self.n_episodes: {self._n_episodes}")
        logger.debug(f"Reached step limit: {self.reached_step_limit}")
        logger.debug(f"Reached episode limit: {self.reached_episode_limit}")
        logger.debug(f"Reached episode length limit: {self.reached_episode_length_limit}")
        
        if self.reached_episode_limit or self.reached_step_limit:
            logger.debug("Done iterating, closing the env.")
            self.close()

    @property
    def reached_step_limit(self) -> bool:
        if self.max_steps is None:
            return False
        return self._n_steps >= self.max_steps

    @property
    def reached_episode_limit(self) -> bool:
        if self.max_episodes is None:
            return False
        return self._n_episodes >= self.max_episodes

    @property
    def reached_episode_length_limit(self) -> bool:
        if self.max_steps_per_episode is None:
            return False
        return self._n_steps_in_episode >= self.max_steps_per_episode

    @property
    def _done_is_true(self) -> bool:
        """Returns wether self._done is True.
        
        This will always return False if the wrapped env is a VectorEnv,
        regardless of if the some of the values in the self._done array are
        true. This is because the VectorEnvs already reset the underlying envs
        when they have done=True.

        Returns
        -------
        bool
            Wether the episode is considered "done" based on self._done. 
        """
        if isinstance(self._done, bool):
            return self._done
        if isinstance(self.env.unwrapped, VectorEnv):
            # VectorEnvs reset themselves, so we consider the "_done" as False,
            # regarless
            return False
        if isinstance(self._done, Tensor) and not self._done.shape:
            return bool(self._done)
        raise RuntimeError(f"'done' should be a single boolean, but got "
                           f"{self._done} of type {type(self._done)})")

        raise RuntimeError(f"Can't tell if we're done: self._done={self._done}")

    def reset(self, **kwargs) -> ObservationType:
        self._observation = self.env.reset(**kwargs)
        self._reset = True
        self._n_steps_in_episode = 0
        # self._n_episodes += 1
        return self._observation

    def close(self) -> None:
        # This will stop the iterator on the next step.
        # self.max_steps = 0
        self._closed = True
        self._action = None
        self._observation = None
        self._reward = None
        super().close()

    def __len__(self) -> Optional[int]:
        if self.max_steps is None:
            raise RuntimeError(f"The dataset has no length when max_steps is None.")
        return self.max_steps

# from common.transforms import Compose
# from collections.abc import Iterable as _Iterable
# from .utils import has_wrapper

# class TransformEnvDatasetItem(gym.Wrapper, IterableDataset):
#     def __init__(self, env: gym.Env, f: Callable[[EnvDatasetItem], Any]):
#         assert has_wrapper(env, EnvDataset), f"Can only be applied on EnvDataset environments!"
#         super().__init__(env)
#         if isinstance(f, list) and not callable(f):
#             f = Compose(f)
#         self.f: Callable[[EnvDatasetItem], Any] = f

#     def __iter__(self):
#         # TODO: Should we also use apply self on the items?
#         for item in iter(self.env):
#             assert isinstance(item, EnvDatasetItem)
#             yield self.f(item)
#         # return iter(self.env)

#     def __next__(self):
#         item: EnvDatasetItem = next(self.env)
#         obs, done, info = item
#         return self.f(item)
