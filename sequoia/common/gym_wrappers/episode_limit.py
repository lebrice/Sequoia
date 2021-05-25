# IDEA: Limit the total number of episodes, even in vectorized
# environments!
import warnings
from typing import List, Union

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
        self._episode_counter: int = -1 # -1 to account for the initial reset.
        self._done: Union[bool, Sequence[bool]] = False
        if self.is_vectorized:
            self._done = np.zeros(self.env.num_envs, dtype=bool)

    def episode_count(self) -> int:
        return self._episode_counter
    
    def reset(self):
        obs = super().reset()
        if self._episode_counter >= self._max_episodes:
            raise ClosedEnvironmentError(f"Env reached max number of episodes ({self._max_episodes})")

        if not self.is_vectorized:
            # Increment every time for non-vectorized env, or just once for
            # VectorEnvs.
            self._episode_counter += 1
        elif self._episode_counter == -1:
            self._episode_counter = 0
        else:
            # Resetting all envs.
            n_unfinished_envs: int = (self._done == False).sum()
            self._episode_counter += n_unfinished_envs
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if self.is_vectorized:
            self._episode_counter += (done == True).sum()
        else:
            # NOTE: We don't increment the episode counter based on `done` here
            # with non-vectorized environments. Instead, we cound the number of
            # calls to the `reset()` method.
            if done:
                self._episode_counter += 1
        return obs, reward, done, info


class EpisodeLimit(EpisodeCounter, MayCloseEarly):
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

    def reset(self):
        obs = super().reset()
        
        if self._episode_counter > self._max_episodes:
            # assert self.is_closed()
            raise ClosedEnvironmentError(f"Env reached max number of episodes ({self.max_episodes})")

        if self.is_vectorized:
            n_unfinished_envs: int = (~self._done).sum()
            if self._episode_counter != 0 and n_unfinished_envs:
                # Wasting some steps in unfinished environments!
                from gym.utils import colorize
                w = UserWarning(
                    f"Calling .reset() on a VectorEnv resets all the envs, "
                    f"ending episodes prematurely. This env has a limit of "
                    f"{self._max_episodes} episodes in total, so by calling "
                    f"reset() here, you could be wasting {n_unfinished_envs} "
                    f"episodes from your budget!"
                )
                warnings.warn(colorize(f"WARN: {w}", 'yellow'))
                self._episode_counter += n_unfinished_envs

        logger.debug(f"(episode {self._episode_counter}/{self._max_episodes})")
        if self._episode_counter >= self._max_episodes:
            # logger.warning("Beware, entering last episode")
            self.close()
        return self.env.reset()

    def step(self, action):
        if self._is_closed:
            if self._episode_counter >= self._max_episodes:
                raise ClosedEnvironmentError(f"Env reached max number of episodes ({self._max_episodes})")
            raise ClosedEnvironmentError("Can't step through closed env.")

        obs, reward, done, info = self.env.step(action)

        if self.is_vectorized:
            self._episode_counter += (done == True).sum()
        else:
            # NOTE: We don't increment the episode counter based on `done` here
            # with non-vectorized environment. Instead, we cound the number of
            # calls to the `reset()` method.
            if done:
                self._episode_counter += 1

        if self._episode_counter >= self._max_episodes:
            self.close()
        
        return obs, reward, done, info

    