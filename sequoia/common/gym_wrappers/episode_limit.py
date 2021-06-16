# IDEA: Limit the total number of episodes, even in vectorized
# environments!
import warnings
from typing import List, Union, Sequence
from gym.utils import colorize

import gym
import numpy as np
from gym.error import ClosedEnvironmentError
from gym.vector import VectorEnv

from sequoia.utils import get_logger

from .utils import IterableWrapper, MayCloseEarly

logger = get_logger(__file__)


class EpisodeCounter(IterableWrapper):
    """ Closes the environment when a given number of episodes is performed.
    
    NOTE: This also applies to vectorized environments, i.e. the episode counter
    is incremented for when every individual environment reaches the end of an
    episode.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env=env)
        self.is_vectorized = isinstance(env.unwrapped, VectorEnv)
        self._episode_counter: int = 0  # -1 to account for the initial reset?
        self._done: Union[bool, Sequence[bool]] = False
        if self.is_vectorized:
            self._done = np.zeros(self.env.num_envs, dtype=bool)
        self._initial_reset: bool = False

    def episode_count(self) -> int:
        return self._episode_counter

    def reset(self):
        obs = super().reset()

        if self._episode_counter >= self._max_episodes:
            raise ClosedEnvironmentError(
                f"Env reached max number of episodes ({self._max_episodes})"
            )

        if self.is_vectorized:
            if not self._initial_reset:
                self._initial_reset = True
                self._episode_counter = 0
            else:
                # Resetting all envs.
                n_unfinished_envs: int = (self._done == False).sum()
                self._episode_counter += n_unfinished_envs
                self._done[:] = False
        else:
            # Increment every time for non-vectorized env, or just once for
            # VectorEnvs.
            self._episode_counter += 1

        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if self.is_vectorized:
            self._episode_counter += (done == True).sum()
        else:
            # NOTE: We don't increment the episode counter based on `done` here
            # with non-vectorized environments. Instead, we cound the number of
            # calls to the `reset()` method.
            pass
            # if done:
            #     self._episode_counter += 1
        return obs, reward, done, info


class EpisodeLimit(EpisodeCounter):
    """ Closes the environment when a given number of episodes is performed.
    
    NOTE: This also applies to vectorized environments, i.e. the episode counter
    is incremented for when every individual environment reaches the end of an
    episode.
    """

    def __init__(self, env: gym.Env, max_episodes: int):
        super().__init__(env=env)
        self._max_episodes = max_episodes

    @property
    def max_episodes(self) -> int:
        return self._max_episodes

    def closed_error_message(self) -> str:
        """ Return the error message to use when attempting to use the closed env.
        
        This can be useful for wrappers that close when a given condition is reached,
        e.g. a number of episodes has been performed, which could return a more relevant
        message here.
        """
        return f"Env reached max number of episodes ({self.max_episodes})"

    def reset(self):
        # NOTE: MayCloseEarly.reset() will raise a ClosedEnvironmentError if
        # self.is_closed() is True, which will always be the case if we exceed the
        # limit.
        obs = super().reset()
        assert not self.is_closed()
        
        if self.is_vectorized:
            n_unfinished_envs: int = (~self._done).sum()
            if self._episode_counter != 0 and n_unfinished_envs:
                # Wasting some steps in unfinished environments!
                w = UserWarning(
                    f"Calling .reset() on a VectorEnv resets all the envs, "
                    f"ending episodes prematurely. This env has a limit of "
                    f"{self._max_episodes} episodes in total, so by calling "
                    f"reset() here, you could be wasting {n_unfinished_envs} "
                    f"episodes from your budget!"
                )
                warnings.warn(colorize(f"WARN: {w}", "yellow"))

        logger.debug(f"Starting episode  {self._episode_counter}/{self._max_episodes})")
        if self._episode_counter == self._max_episodes:
            logger.warning("Beware, entering last episode")
        return obs

    def __iter__(self):
        return super().__iter__()

    def step(self, action):
        if self.is_closed():
            if self._episode_counter >= self._max_episodes:
                raise ClosedEnvironmentError(
                    f"Env reached max number of episodes ({self._max_episodes})"
                )
            raise ClosedEnvironmentError("Can't step through closed env.")

        obs, reward, done, info = super().step(action)

        if self.is_vectorized:
            if any(done) and self._episode_counter >= self.max_episodes:
                logger.info(
                    f"Closing the envs since we reached the max number of episodes."
                )
                self.close()
                done[:] = True
        else:
            if done and self._episode_counter == self._max_episodes:
                logger.info(
                    f"Closing the env since we reached the max number of episodes."
                )
                self.close()

        return obs, reward, done, info

