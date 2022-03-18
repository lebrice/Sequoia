""" IDEA: same as EpisodeLimit, for for the number of total observations.
"""

import gym
from gym.error import ClosedEnvironmentError

from sequoia.utils import get_logger

from .utils import IterableWrapper

logger = get_logger(__name__)


class ObservationLimit(IterableWrapper):
    """Closes the env when `max_steps` steps have been performed *in total*.

    For vectorized environments, each step consumes up to `num_envs` from this
    total budget, i.e. the step counter is incremented by the batch size at
    each step.
    """

    def __init__(self, env: gym.Env, max_steps: int):
        super().__init__(env=env)
        self._max_obs = max_steps
        self._obs_counter: int = 0
        self._initial_reset = False
        self._is_closed: bool = False

    def reset(self):
        if self._is_closed:
            if self._obs_counter >= self._max_obs:
                raise ClosedEnvironmentError(
                    f"Env reached max number of observations ({self._max_obs})"
                )
            raise ClosedEnvironmentError("Can't step through closed env.")

        # Resetting actually gives you an observation, so we count it here.
        self._obs_counter += self.env.num_envs if self.is_vectorized else 1
        logger.debug(f"(observation {self._obs_counter}/{self._max_obs})")

        obs = self.env.reset()

        if self._obs_counter >= self._max_obs:
            self.close()

        return obs

    @property
    def is_closed(self) -> bool:
        return self._is_closed

    def step(self, action):
        if self._is_closed:
            if self._obs_counter >= self._max_obs:
                raise ClosedEnvironmentError(
                    f"Env reached max number of observations ({self._max_obs})"
                )
            raise ClosedEnvironmentError("Can't step through closed env.")

        obs, reward, done, info = self.env.step(action)

        self._obs_counter += self.env.num_envs if self.is_vectorized else 1
        logger.debug(f"(observation {self._obs_counter}/{self._max_obs})")

        # BUG: If we dont use >=, then iteration with EnvDataset doesn't work.
        if self._obs_counter >= self._max_obs:
            self.close()

        return obs, reward, done, info

    def close(self):
        self.env.close()
        self._is_closed = True
