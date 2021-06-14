""" Creates an IterableDataset from a Gym Environment. 
"""
from collections import abc as collections_abc
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)
import warnings
import gym
import numpy as np
from torch import Tensor
from torch.utils.data import IterableDataset

from sequoia.utils.logging_utils import get_logger

from .batch_env import VectorEnv
from .utils import (
    ActionType,
    ObservationType,
    RewardType,
    StepResult,
    MayCloseEarly as CloseableWrapper,
)

# from sequoia.settings.base.objects import Observations, Rewards, Actions
logger = get_logger(__file__)


Item = TypeVar("Item")


class EnvDataset(
    CloseableWrapper,
    IterableDataset,
    Generic[ObservationType, ActionType, RewardType, Item],
    Iterable[Item],
):
    """ Wrapper that exposes a Gym environment as an IterableDataset.

    This makes it possible to iterate over a gym env with an Active DataLoader.
    
    One pass through __iter__ is one episode. The __iter__ method can be called
    at most `max_episodes` times.
    """

    def __init__(
        self,
        env: gym.Env,
        max_steps: Optional[int] = None,
        max_episodes: Optional[int] = None,
        max_steps_per_episode: Optional[int] = None,
    ):
        super().__init__(env=env)
        if isinstance(env.unwrapped, VectorEnv):
            if not max_steps_per_episode:
                warnings.warn(
                    UserWarning(
                        "Iterations through the dataset (episodes) could be "
                        "infinitely long, since the env is a VectorEnv and "
                        "max_steps_per_episode wasn't given!"
                    )
                )

        # Maximum number of episodes per iteration.
        self.max_episodes = max_episodes
        # Maximum number of steps per iteration.
        self.max_steps = max_steps
        self.max_steps_per_episode = max_steps_per_episode

        # Number of steps performed in the current episode.
        self.n_steps_in_episode_: int = 0

        # Total number of steps performed so far.
        self.n_steps_: int = 0
        # Number of episodes performed in the environment.
        # Starts at -1 so the initial was_reset doesn't count as the end of an episode.
        self.n_episodes_: int = 0
        # Number of times the `send` method was called.
        self.n_sends_: int = 0

        self.observation_: Optional[ObservationType] = None
        self.action_: Optional[ActionType] = None
        self.reward_: Optional[RewardType] = None
        self.done_: Optional[Union[bool, Sequence[bool]]] = None
        self.info_: Optional[Union[Dict, Sequence[Dict]]] = None

        self.closed_: bool = False
        self.reset_: bool = False

        self.current_step_result_: StepResult = None
        self.previous_step_result_: StepResult = None

    def reset_counters(self):
        self.n_steps_ = 0
        self.n_episodes_ = 0
        self.n_sends_ = 0
        self.n_steps_in_episode_ = 0

    def observation(self, observation):
        return observation

    def action(self, action):
        return action

    def reward(self, reward):
        return reward

    def step(self, action) -> StepResult:
        if self.closed_ or self.is_closed():
            if self.reached_episode_limit:
                raise gym.error.ClosedEnvironmentError(
                    f"Env has already reached episode limit ({self.max_episodes}) and is closed."
                )
            elif self.reached_step_limit:
                raise gym.error.ClosedEnvironmentError(
                    f"Env has already reached step limit ({self.max_steps}) and is closed."
                )
            else:
                raise gym.error.ClosedEnvironmentError(f"Can't call step on closed env. ({self.n_steps_})")
        # Here we add calls to the (potentially overwritten) 'observation',
        # 'action' and 'reward' methods.
        action = self.action(action)
        if isinstance(action, Tensor) and action.requires_grad:
            action = action.detach()
        observation, reward, done, info = self.env.step(action)
        observation = self.observation(observation)
        reward = self.reward(reward)
        self.n_steps_ += 1
        self.n_steps_in_episode_ += 1

        result = StepResult(observation, reward, done, info)
        self.previous_step_result_ = self.current_step_result_
        self.current_step_result_ = result
        return result

    def __next__(
        self,
    ) -> Tuple[
        ObservationType, Union[bool, Sequence[bool]], Union[Dict, Sequence[Dict]]
    ]:
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
        # logger.debug(f"__next__ is being called at step {self.n_steps_}.")

        if self.closed_:
            raise gym.error.ClosedEnvironmentError("Env is closed.")

        if self.reached_episode_limit:
            logger.debug("Reached episode limit, raising StopIteration.")
            raise StopIteration
        if self.reached_step_limit:
            logger.debug("Reached step limit, raising StopIteration.")
            raise StopIteration
        if self.reached_episode_length_limit:
            logger.debug("Reached episode length limit, raising StopIteration.")
            raise StopIteration

        if not self.reset_:
            raise gym.error.ResetNeeded(
                "Need to reset the env before you can call __next__"
            )

        if self.action_ is None:
            raise RuntimeError(
                "You have to send an action using send() between every observation."
            )
        if hasattr(self.action_, "detach"):
            self.action_ = self.action_.detach()
        self.observation_, self.reward_, self.done_, self.info_ = self.step(
            self.action_
        )
        return self.observation_

    def send(self, action: ActionType) -> RewardType:
        """ Sends an action to the environment, returning a reward.
        This can raise the same errors as calling __next__, namely,
        StopIteration, ResetNeeded,  raise an error when if not called without
        """
        assert action is not None, "Don't send a None action!"
        self.action_ = action
        self.observation_ = self.__next__()
        self.n_sends_ += 1
        return self.reward_

    def __iter__(self) -> Iterable[ObservationType]:
        """Iterator for an episode in the environment, which uses the 'active
        dataset' style with __iter__ and send.

        TODO: BUG: Wrappers applied on top of the EnvDataset won't have an
        effect on the values yielded by this iterator. Currently trying to fix
        this inside the IterableWrapper base class, but it's not that simple.      
        
        TODO: To allow wrappers to also be iterable, we need to rename all the
        "private" attributes to "public" names, so that they can call something
        like:
        type(self.env).__iter__(self) (from within the wrapper).  
        
        Yields
        -------
        Observations
            Observations from the environment.

        Raises
        ------
        RuntimeError
            [description]
        """
        if self.closed_ or self.is_closed():
            if self.reached_episode_limit:
                raise gym.error.ClosedEnvironmentError(
                    f"Env has already reached episode limit ({self.max_episodes}) and is closed."
                )
            elif self.reached_step_limit:
                raise gym.error.ClosedEnvironmentError(
                    f"Env has already reached step limit ({self.max_steps}) and is closed."
                )
            else:
                raise gym.error.ClosedEnvironmentError(
                    f"Env is closed, can't iterate over it."
                )

        # First step reset automatically before iterating, if needed.
        if not self.reset_:
            self.observation_ = self.reset()

        self.done_ = False
        self.action_ = None
        self.reward_ = None

        assert self.observation_ is not None
        # Yield the first observation_.
        # TODO: What do we want to yield, actually? Just observations?
        yield self.observation_

        # logger.debug(f"episode {self.n_episodes_}/{self.max_episodes}")

        while not any(
            [
                self.done_is_true(),
                self.reached_step_limit,
                self.reached_episode_length_limit,
                self.is_closed(),
            ]
        ):
            # logger.debug(f"step {self.n_steps_}/{self.max_steps},  (episode {self.n_episodes_})")

            # Set those to None to force the user to call .send()
            self.action_ = None
            self.reward_ = None
            yield self.observation_

            if self.action_ is None:
                raise RuntimeError(
                    f"You have to send an action using send() between every "
                    f"observation. (env = {self})"
                )

        # Force the user to call reset() between episodes.
        self.reset_ = False
        self.n_episodes_ += 1

        # logger.debug(f"self.n_steps: {self.n_steps_} self.n_episodes: {self.n_episodes_}")
        # logger.debug(f"Reached step limit: {self.reached_step_limit}")
        # logger.debug(f"Reached episode limit: {self.reached_episode_limit}")
        # logger.debug(f"Reached episode length limit: {self.reached_episode_length_limit}")

        if self.reached_episode_limit or self.reached_step_limit:
            logger.debug("Done iterating, closing the env.")
            self.close()

    @property
    def reached_step_limit(self) -> bool:
        if self.max_steps is None:
            return False
        return self.n_steps_ >= self.max_steps

    @property
    def reached_episode_limit(self) -> bool:
        if self.max_episodes is None:
            return False
        return self.n_episodes_ >= self.max_episodes

    @property
    def reached_episode_length_limit(self) -> bool:
        if self.max_steps_per_episode is None:
            return False
        return self.n_steps_in_episode_ >= self.max_steps_per_episode

    # @property
    def done_is_true(self) -> bool:
        """Returns wether self.done_ is True.
        
        This will always return False if the wrapped env is a VectorEnv,
        regardless of if the some of the values in the self.done_ array are
        true. This is because the VectorEnvs already reset the underlying envs
        when they have done=True.

        Returns
        -------
        bool
            Wether the episode is considered "done" based on self.done_. 
        """
        if isinstance(self.done_, bool):
            return self.done_
        if isinstance(self.env.unwrapped, VectorEnv):
            # VectorEnvs reset themselves, so we consider the "_done" as False,
            # regarless
            return False
        if isinstance(self.done_, Tensor) and not self.done_.shape:
            return bool(self.done_)
        raise RuntimeError(
            f"'done' should be a single boolean, but got "
            f"{self.done_} of type {type(self.done_)})"
        )

        raise RuntimeError(f"Can't tell if we're done: self.done_={self.done_}")

    def reset(self, **kwargs) -> ObservationType:
        observation = self.env.reset(**kwargs)
        self.observation_ = self.observation(observation)
        self.reset_ = True
        self.n_steps_in_episode_ = 0
        # self.n_episodes_ += 1
        return self.observation_

    def close(self) -> None:
        # This will stop the iterator on the next step.
        # self.max_steps = 0
        self.closed_ = True
        self.action_ = None
        self.observation_ = None
        self.reward_ = None
        super().close()

    # TODO: calling `len` on an RL environment probably shouldn't work! (it should
    # behave the same exact way as an IterableDataset)

    def __len__(self) -> Optional[int]:
        if self.max_steps is None:
            raise RuntimeError(f"The dataset has no length when max_steps is None.")
        return self.max_steps
