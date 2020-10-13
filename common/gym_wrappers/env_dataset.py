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
from settings.base.objects import Observations, Rewards, Actions
logger = get_logger(__file__)


def state_transition(observation, action: Any, next_observation, reward):
    """ Determines what an 'item' of the dataset below will look like.

    By default, only keeps the observations of the current step.
    """
    # TODO: Clarify this here...
    #       Observations      |     Rewards ?
    # (observations, actions) -> (observations, rewards) ?
    
    # return previous_step_results[0], current_step_results[1]
    return (observation, action, next_observation), reward


def random_policy(observation: ObservationType, action_space: gym.Space):
    """ Policy that takes a random action. """
    return action_space.sample()


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
                 policy: Callable[[ObservationType, gym.Space], ActionType] = None,
                 max_episodes: Optional[int] = None,
                 max_steps: Optional[int] = None,
                 dataset_item_type: Callable[[ObservationType, ActionType, ObservationType, RewardType],
                                             Item] = state_transition,
                 ):
        super().__init__(env=env)
        if isinstance(env, AsyncVectorEnv):
            assert not max_episodes, (
                "TODO: No notion of 'episode' when using a batched environment!"
            )
        self.dataset_item_type = dataset_item_type or state_transition
        # Maximum number of episodes per iteration.
        self.max_episodes = max_episodes
        # Maximum number of steps per iteration.
        self.max_steps = max_steps
        # Callable to use when we're missing an action.
        self.policy = policy
        
        # Total number of steps performed so far.
        self._n_steps: int = 0
        # Number of episodes performed in the environment.
        self._n_episodes: int = 0
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
        self.policy = policy

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
        
        if self._closed:
            raise gym.error.ClosedEnvironmentError("Env is closed.")
        if self.reached_episode_limit:
            logger.debug("Reached episode limit, raising StopIteration.")
            raise StopIteration
        if self.reached_step_limit:
            logger.debug("Reached step limit, raising StopIteration.")
            raise StopIteration
        if not self._reset:
            raise gym.error.ResetNeeded("Need to reset the env before you can call __next__")
        assert self._action is not None
        self._observation, self._reward, self._done, self._info = self.step(self._action)
        return self._observation

    #     # NOTE: See the docstring of `GymDataLoader` for an explanation of why
    #     # this doesn't return the same thing as `step()`.

    #     # self._action was passed through 'send', now we can give back the observations
    #     # of the previous step, and then update the env.
    #     if self._current_step_result is None:
    #         # This means we're at the start of an episode.
    #         self._action = self.action_space.sample()
    #         self._previous_step_result = self.step(self._action)

    #     # Update the env for the next step.
    #     self._previous_step_result = self._current_step_result
    #     self._current_step_result = self.step(self._action)

    #     item = self.env_dataset_item(self._current_step_result)        
    #     # Delete the action, to force the user to pass one to send or to use a
    #     # default policy.
    #     self._action = None
    #     return item

    def send(self, action: ActionType) -> RewardType:
        """ Sends an action to the environment, returning a reward.
        This will raise an error when if not called without
        """
        # assert False, (
        #     "TODO: work in progress, if you're gonna pretend this is an "
        #     "'active' environment, then use the gym API for now."
        # )
        assert action is not None, "Don't send a None action!"
        self._action = action
        self.__next__()
        # self._observations, self._rewards, self._done, self._info = self.step(self._action)
        self._n_sends += 1
        return self._reward

    def __iter__(self) -> Iterable[Tuple[ObservationType,
                                         Union[bool, Sequence[bool]],
                                         Union[Dict, Sequence[Dict]]]]:
        """Iterator for an episode of a gym environment.

        TODO: Need to think a bit harder about how to set this up..
        """

        if self._closed:
            raise gym.error.ClosedEnvironmentError("Env has already reached limit and is closed.")
        
        if self.policy:
            return self.iterator_with_policy()
        else:
            return self.iterator_with_send()

    def iterator_with_send(self) -> Iterable[Observations]:
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
        
        if self._closed:
            raise gym.error.ClosedEnvironmentError("Can't iterate over closed Env.")

        if not self._reset:
            self._observation = self.reset()
        self._done = False
        self._action = None
        self._reward = None
        assert self._observation is not None
        yield self._observation
        
        logger.debug(f"episode {self._n_episodes}/{self.max_episodes}")

        while not self._done and not self.reached_step_limit:
            logger.debug(f"step {self._n_steps}/{self.max_steps}, ")
            
            # Set those to None to force the user to call .send()
            self._action = None
            self._reward = None
            yield self._observation

            if self._action is None:
                raise RuntimeError(
                    "You have to send an action using send() between every "
                    "observation (since there is no policy)."
                )

            if not isinstance(self._done, bool):
                if any(self._done):
                    raise RuntimeError(
                        "done should either be a bool or all be false, since "
                        "we can't do partial resets."
                    )
                self._done = False

        # Force the user to call reset() between episodes.
        self._reset = False
        self._n_episodes += 1
        
        logger.debug(f"self.n_steps: {self._n_steps} self.n_episodes: {self._n_episodes}")
        logger.debug(f"Reached step limit: {self.reached_step_limit}")
        logger.debug(f"Reached episode limit: {self.reached_episode_limit}")
        
        if self.reached_episode_limit or self.reached_step_limit:
            logger.debug("Done iterating, closing the env.")
            self.close()

    def iterator_with_policy(self) -> Iterable[Item]:
        """Iterate when using a policy, yielding 'items' which could be state
        transitions.

        Returns
        -------
        Iterable[Item]
            [description]

        Yields
        -------
        Iterator[Iterable[Item]]
            [description]

        Raises
        ------
        RuntimeError
            [description]
        """
        while not (self.reached_episode_limit or self.reached_step_limit):
            previous_observations = self.reset()
            done = False
            
            logger.debug(f"episode {self._n_episodes}/{self.max_episodes}")
            
            while not done and not self.reached_step_limit:
                logger.debug(f"step {self._n_steps}/{self.max_steps}, ")
                # Get the batch of actions using the policy.
                actions = self.policy(previous_observations, self.action_space)
                
                observations, rewards, done, info = self.step(actions)
                
                # TODO: Need to figure out what to yield here..
                yield self.dataset_item_type(previous_observations, actions, observations, rewards)
                
                previous_observations = observations
                
                if not isinstance(done, bool):
                    if any(done):
                        raise RuntimeError(
                            "done should either be a bool or always false, since "
                            "we can't do partial resets."
                        )
                    done = False

            self._n_episodes += 1
        self._reset = False
        logger.debug(f"self.n_steps: {self._n_steps} self.n_episodes: {self._n_episodes}")
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
        self._observation = self.env.reset(**kwargs)
        self._reset = True
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
