""" Idea: a simple wrapper that counts the number of steps and episodes, with
optional arguments used to limit the number of steps until the env is done.
"""

from typing import (Dict, Generator, Generic, Iterable, List, Optional,
                    Sequence, Tuple, Type, TypeVar, Union)

import gym
from torch.utils.data import IterableDataset

from utils.logging_utils import get_logger

from .batch_env import BatchEnv

ObservationType = TypeVar("ObservationType")
ActionType = TypeVar("ActionType")
RewardType = TypeVar("RewardType")

logger = get_logger(__file__)

# TODO: @lebrice Create a wrapper that stores the last state in the `info` dict.
# depending on the `done` value as well.


class EnvDataset(gym.Wrapper, IterableDataset, Generic[ObservationType, ActionType, RewardType]):
    """ Wrapper that exposes a Gym environment as an IterableDataset.

    This makes it possible to iterate over a gym env with an Active DataLoader.
    """

    def __init__(self,
                 env: gym.Env,
                 max_episodes: Optional[int] = None,
                 max_steps: Optional[int] = None,
                 ):
        super().__init__(env=env)
        if isinstance(env, BatchEnv):
            assert not max_episodes, (
                "TODO: No notion of 'episode' when using a batched environment!"
            )

        # Maximum number of episodes to perform in the environment.
        self.max_episodes = max_episodes
        # Maximum number of steps to perform in the environment.
        self.max_steps = max_steps
        # Number of steps performed in the environment.
        self.n_steps: int = 0
        # Number of times the `send` method was called, i.e. number of actions
        # taken in the environment.
        self.n_actions: int = 0
        # Number of episodes performed in the environment.
        # Starts at -1 so that an initial reset brings it to 0.
        self.n_episodes: int = -1
        # Number of samples yielded by the iterator so far.
        self.n_pulled: int = 0
        self._observation: Optional[ObservationType] = None 
        self._action: Optional[ActionType] = None
        self._reward: Optional[RewardType] = None
        self._done: Optional[Union[bool, Sequence[bool]]] = None
        self._info: Optional[Union[Dict, Sequence[Dict]]] = None

    def step(self, action) -> Tuple[ObservationType,
                                    RewardType,
                                    Union[bool, Sequence[bool]],
                                    Union[Dict, Sequence[Dict]]]:
        self._action = action
        self._observation, self._reward, self._done, self._info = super().step(self._action)
        self.n_steps += 1
        assert self._observation is not None
        assert self._reward is not None
        assert self._done is not None
        assert self._info is not None
        return self._observation, self._reward, self._done, self._info

    def __next__(self) -> Tuple[ObservationType,
                                Union[bool, Sequence[bool]],
                                Union[Dict, Sequence[Dict]]]:
        # NOTE: See the docstring of `GymDataLoader` for an explanation of why
        # this doesn't return the same thing as `step()`.
        return self._observation, self._done, self._info
        return self.step(self._action)

    def __iter__(self) -> Iterable[Tuple[ObservationType,
                                         Union[bool, Sequence[bool]],
                                         Union[Dict, Sequence[Dict]]]]:
        # Reset the env if it hasn't been called before iterating.
        if self.n_episodes == -1:
            logger.debug(f"Resetting env because it hasn't been done before iterating.")
            self.reset()

        while not (self.reached_episode_limit or self.reached_step_limit):
            # Perform an episode.
            done = self._done if isinstance(self._done, bool) else False

            while not done or self.reached_step_limit:
                # TODO: @lebrice Isn't there something fishy going on here? I'm
                # not sure that we're giving back the right reward for the right
                # action and observation?

                # TODO: This should be a 'push' model, steps should occur when
                # the action is received, the corresponding reward be
                # immediately returned, and the next yield statement in the
                # iterator should give back the rest ? (Need to figure this out)

                logger.debug(f"(step={self.n_steps}, n_pulled={self.n_pulled}, n_actions={self.n_actions}, n_episodes={self.n_episodes})")
                if self.n_pulled > self.n_actions:
                    raise RuntimeError(
                        "You need to send an action using the `send` method "
                        "every time you get a value from the dataset! "
                        "Otherwise, you can also pass in a policy to use when "
                        "an action isn't given. \n"
                    )
                action = yield self.__next__()
                self.n_pulled += 1
                
                if action is not None:
                    raise NotImplementedError(
                        "Send actions to the env using the `send` method on "
                        "the env, not on the iterator itself!"
                    )
                if isinstance(self.env, BatchEnv) and any(self._done):
                    # Only reset the envs that need to be reset, in this case.
                    self.env.partial_reset(self._done)

            self.n_episodes += 1
            self.reset()
        self.close()

    def send(self, action: ActionType) -> RewardType:
        assert action is not None, "Don't send a None action!"
        self.n_actions += 1
        self.step(action)
        assert self._observation is not None
        assert self._done is not None
        assert self._reward is not None
        return self._reward

    @property
    def reached_step_limit(self) -> bool:
        if self.max_steps is not None:
            return self.n_steps >= self.max_steps
        return False
    
    @property
    def reached_episode_limit(self) -> bool:
        if self.max_episodes is not None:
            return self.n_episodes >= self.max_episodes
        return False

    def reset(self, **kwargs) -> ObservationType:
        self._observation = super().reset(**kwargs)
        self._reward = 0
        self._done = False
        self._info = {}
        self.n_episodes += 1
        return self._observation

    def close(self) -> None:
        # This will stop the iterator on the next step.
        self.max_steps = 0
        super().close()
