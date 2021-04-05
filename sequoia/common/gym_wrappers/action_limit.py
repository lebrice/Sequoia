""" IDEA: same as ObservationLimit, for for the number of total actions (steps).
"""
import gym
from gym.error import ClosedEnvironmentError
from gym.vector import VectorEnv
from sequoia.utils import get_logger

from .utils import IterableWrapper

logger = get_logger(__file__)


class ActionCounter(IterableWrapper):
    """ Wrapper that counts the total number of actions performed so far.
    (including those in the individual environments when wrapping a VectorEnv.)
    """

    def __init__(self, env: gym.Env):
        super().__init__(env=env)
        self.is_vectorized = isinstance(env.unwrapped, VectorEnv)
        self._action_counter: int = 0

    def action_count(self) -> int:
        return self._action_counter

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._action_counter += self.env.num_envs if self.is_vectorized else 1
        return obs, reward, done, info


class ActionLimit(ActionCounter):
    """ Closes the env when `max_steps` actions have been performed *in total*.

    For vectorized environments, each step consumes up to `num_envs` from this
    total budget, i.e. the step counter is incremented by the batch size at
    each step.
    """

    def __init__(self, env: gym.Env, max_steps: int):
        super().__init__(env=env)
        self.is_vectorized = isinstance(env.unwrapped, VectorEnv)

        self._max_steps = max_steps
        self._initial_reset = False
        self._is_closed: bool = False

    @property
    def max_steps(self) -> int:
        return self._max_steps

    def step(self, action):
        if self._action_counter >= self._max_steps:
            raise ClosedEnvironmentError(
                f"Env reached max number of steps ({self._max_steps})"
            )

        obs, reward, done, info = super().step(action)
        # logger.debug(f"(step {self._action_counter}/{self._max_steps})")

        # BUG: If we dont use >=, then iteration with EnvDataset doesn't work.
        if self._action_counter >= self._max_steps:
            self.close()

        return obs, reward, done, info
