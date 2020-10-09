""" Creates an IterableDataset from a Gym Environment. 
"""
from collections import abc as collections_abc
from typing import (Any, Callable, Dict, Generator, Generic, Iterable, List,
                    Optional, Sequence, Tuple, Type, TypeVar, Union)

import gym
from torch.utils.data import IterableDataset

from utils.logging_utils import get_logger

from .batch_env import AsyncVectorEnv
from .utils import ActionType, ObservationType, RewardType, StepResult

logger = get_logger(__file__)

# def on_missing_action(self,
#                       observation: ObservationType,
#                       done: Union[bool, Sequence[bool]],
#                       info: Union[Dict, Sequence[Dict]]) -> ActionType:

def env_dataset_item(step_result):
    """ Chooses what is kept from the step result and considered an 'item'.
    for the iterable below.
    
    By default, only keeps the observations.
    """
    return step_result[0]


# We could maybe use this wrapper too?
from gym.wrappers import TimeLimit


def random_policy(observation: ObservationType, action_space: gym.Space):
    """ Policy that takes a random action. """
    return action_space.sample()


class EnvDataset(gym.Wrapper, IterableDataset, Generic[ObservationType, ActionType, RewardType]):
    """ Wrapper that exposes a Gym environment as an IterableDataset.

    This makes it possible to iterate over a gym env with an Active DataLoader.
    
    One pass through __iter__ is one episode. The __iter__ method can be called
    at most `max_episodes` times.
    """
    def __init__(self,
                 env: gym.Env,
                 max_episodes: Optional[int] = None,
                 max_steps: Optional[int] = None,
                 default_policy: Callable[[ObservationType, gym.Space], ActionType] = None,
                 dataset_item_type: Callable[[Tuple[Any, Any, Any]], Any] = env_dataset_item,
                 ):
        super().__init__(env=env)
        if isinstance(env, AsyncVectorEnv):
            assert not max_episodes, (
                "TODO: No notion of 'episode' when using a batched environment!"
            )
        self.dataset_item_type = dataset_item_type
        # Maximum number of episodes per iteration.
        self.max_episodes = max_episodes
        # Maximum number of steps per iteration.
        self.max_steps = max_steps
        # Callable to use when we're missing an action.
        self.default_policy = default_policy
        
        # Total number of steps performed so far.
        self._n_steps: int = 0
        # Number of episodes performed in the environment.
        self._n_episodes: int = 0
        # Number of times the `send` method was called.
        self._n_sends: int = 0

        # self._observation: Optional[ObservationType] = None
        self._action: Optional[ActionType] = None
        # self._reward: Optional[RewardType] = None
        self._done: Optional[Union[bool, Sequence[bool]]] = None
        # self._info: Optional[Union[Dict, Sequence[Dict]]] = None

        self._closed: bool = False
        # IDEA: Maybe use the id() of the action/observation to check if we're
        # calling `send` or `next` two times in a row?
        # self._action_id: int = None
        
        self._current_step_result: StepResult = None
        self._previous_step_result: StepResult = None

    def set_policy(self, policy: Callable[[ObservationType, gym.Space], ActionType]) -> None:
        """Set a policy to use when we're missing an action when iterating.
         
        The policy should take an observation and the action space, and return
        an action. The policy will never be used on observations of 'done'
        states.
        
        Parameters
        ----------
        policy : Callable[[EnvDatasetItem, gym.Space], ActionType]
            [description]
        """
        self.default_policy = policy

    def reset_counters(self):
        self._n_steps = 0
        self._n_episodes = 0
        self._n_sends = 0

    def step(self, action) -> StepResult:
        result = StepResult(*self.env.step(action))
        self._n_steps += 1
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
        StopIteration
            When the step limit has been reached. 
        StopIteration
            When the episode limit has been reached.
        RuntimeError
            When an action wasn't passed through 'send', and a default policy
            isn't set.
        """
        logger.debug(f"__next__ is being called at step {self._n_steps}.")
        # NOTE: See the docstring of `GymDataLoader` for an explanation of why
        # this doesn't return the same thing as `step()`.
        if self._closed:
            raise gym.error.ClosedEnvironmentError("Env has already reached limit and is closed.")
        elif self.reached_step_limit:
            logger.debug("Reached step limit, raising StopIteration.")
            raise StopIteration
        elif self.reached_episode_limit:
            logger.debug("Reached episode limit, raising StopIteration.")
            raise StopIteration
        
        assert self._action is not None, "self._action should have been set through 'send'."
        # self._action was passed through 'send', now we can give back the observations
        # of the previous step, and then update the env.
        
        if self._current_step_result is None:
            # This means we're at the start of an episode.
            obs, reward, done, info = self._current_step_result
            item = self.dataset_item_type(obs, done, info)
            # Update the env for the next step.
            self._previous_step_result = self._current_step_result
            self._current_step_result = self.step(self._action)
            # Delete the action, to force the user to pass one to send or to use a
            # default policy.
            self._action = None
              
        obs, reward, done, info = self._current_step_result
        item = self.dataset_item_type(obs, done, info)
        # Update the env for the next step.
        self._previous_step_result = self._current_step_result
        self._current_step_result = self.step(self._action)
        # Delete the action, to force the user to pass one to send or to use a
        # default policy.
        self._action = None
        return item

    def send(self, action: ActionType) -> RewardType:
        """ Sends an action to the environment, returning a reward.
        This will raise an error when if not called without
        """
        assert action is not None, "Don't send a None action!"
        if self._current_step_result is None:
            assert self._n_steps == 0
            # We just reset the env, so we don't have a reward yet!
            self._action = action
            self.__next__()
            return self._current_step_result[1]
        
        if self._action is not None:
            raise RuntimeError("Shouldn't be calling 'send' twice in a row!")
        
        self._action = action
        self._n_sends += 1
        reward = self._current_step_result[1]
        return reward

    def __iter__(self) -> Iterable[Tuple[ObservationType,
                                         Union[bool, Sequence[bool]],
                                         Union[Dict, Sequence[Dict]]]]:
        """Iterator for an episode of a gym environment.

        TODO: To keep things simple, this should perhaps iterate over a single episode?
        TODO: Should we ask the user to reset the env before we can iterate on it? 
        OR, we could make it so if the environment hasn't been reset before
        iterating, the first batch will be the result of `self.env.reset()`.
                
        Returns
        -------
        Iterable[Tuple[ObservationType, Union[bool, Sequence[bool]], Union[Dict, Sequence[Dict]]]]
            [description]

        Yields
        -------
        Iterator[Iterable[Tuple[ObservationType, Union[bool, Sequence[bool]], Union[Dict, Sequence[Dict]]]]]
            [description]

        Raises
        ------
        NotImplementedError
            [description]
        """
        # Yield the initial observation of the episode.
        yield self.reset()

        while not self.end_episode:
            # Perform an episode.
            logger.debug(f"step {self._n_steps}/{self.max_steps}, "
                         f"episode {self._n_episodes}/{self.max_episodes}")
            # logger.debug(f"Before yield")
            action = yield self.__next__()
            # logger.debug(f"After yield")
            
            if action is not None:
                # We can't allow this, because then we'd have no consistent way
                # to return the rewards.
                raise NotImplementedError(
                    "Send actions to the env using the `send` method on "
                    "the env, not on the iterator itself!"
                )
  
        self.n_episodes += 1
        
        logger.debug(f"self.n_steps: {self.n_steps} self.n_episodes: {self.n_episodes}")
        logger.debug(f"Reached step limit: {self.reached_step_limit}")
        logger.debug(f"Reached episode limit: {self.reached_episode_limit}")
        
        if self.reached_episode_limit:
            logger.debug(f"Done iterating, closing the env.")
            self.close()

    @property
    def reached_step_limit(self) -> bool:
        if self.max_steps is not None:
            return self._n_steps >= self.max_steps
        return False

    @property
    def reached_episode_limit(self) -> bool:
        if self.max_episodes is not None:
            return self._n_episodes >= self.max_episodes
        return False

    @property
    def end_episode(self) -> bool:
        """Returns wether we should end the current episode.

        Returns
        -------
        bool
            Wether the episode is done, or if we reach the step limit.
        """
        if isinstance(self._done, bool):
            return self._done
        if isinstance(self._done, collections_abc.Iterable):
            return all(self._done)
        return self.reached_step_limit

    def reset(self, **kwargs) -> ObservationType:
        return self.env.reset(**kwargs)

    def close(self) -> None:
        # This will stop the iterator on the next step.
        # self.max_steps = 0
        self._closed = True
        super().close()

    def __len__(self) -> Optional[int]:
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
