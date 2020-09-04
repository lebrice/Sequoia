from functools import wraps
from typing import Any, Callable, List, Tuple, Union, Any

import gym
import numpy as np
import torch
from torch import Tensor

from conftest import xfail
from settings.base import EnvironmentBase
from utils.logging_utils import get_logger

from .gym_dataloader import GymDataLoader
from .gym_dataset import GymDataset

logger = get_logger(__file__)


def check_interaction_with_env(env: Union[GymDataset, GymDataLoader],
                               obs_shape: Tuple[int, ...],
                               reward_shape: Tuple[int, ...],
                               action: Any=1,
                               n_batches: int=5):
    """Get a few batches from the env and make sure they have the desired shape.
    Also sends the given action at each step.
    """
    env.reset()
    for i, x in zip(range(n_batches), env):
        assert isinstance(x, (Tensor, np.ndarray))
        assert x.shape == obs_shape
        reward = env.send(action)
        assert isinstance(reward, (float, np.ndarray, Tensor))
        if isinstance(reward, (np.ndarray, Tensor)):
            assert reward.shape == reward_shape
    env.close()

from typing import Optional
from gym import spaces

class DummyEnvironment(gym.Env):
    """ Dummy environment for testing.
    
    The reward is how close to the target the state is, and the actions are
    either to increment or decrement the counter value.
    """
    def __init__(self, max_value: int = 10):
        self.max_value = max_value
        self.i = 0
        self.reward_range = (0, max_value)
        self.action_space = spaces.Discrete(n=2)
        self.observation_space = spaces.Discrete(n=max_value)

        self.target = max_value // 2

        self.done: bool = False

    def step(self, action: int):
        # The action modifies the state, producing a new state, and you get the
        # reward associated with that transition.
        if action == 0:
            self.i += 1
        elif action == 1:
            self.i -= 1
        self.i %= self.max_value
        self.done = self.i == self.max_value
        reward = abs(self.i - self.target)
        return self.i, reward, self.done, {}

    def reset(self):
        self.i = 0

def test_number_of_steps_taken():
    """TODO: Trying to write a test
    """
    max_value = 10
    base_env = DummyEnvironment(max_value=max_value)
    env: GymDataset[Tensor, int, float] = GymDataset(env=base_env, observe_pixels=False)
    env.reset()
    assert env.step_count == env.send_count == 0
    iterator = iter(env)
    target = max_value // 2
    for i, observation in enumerate(iterator):
        assert env.step_count == i + 1
        assert env.send_count == i

        action = 1
        reward = env.send(action)
        assert env.send_count == i + 1

        assert reward == abs((i+1) - target)

        break
    
    env.close()


def test_wrap_pendulum_pixels():
    env = gym.make("Pendulum-v0")
    bob: GymDataset[Tensor, int, float] = GymDataset(env=env, observe_pixels=True)
    check_interaction_with_env(bob, obs_shape=(500, 500, 3), action=bob.action_space.sample(), reward_shape=(1,))


def test_wrap_cartpole_state():
    env = gym.make("CartPole-v0")
    bob: GymDataset[Tensor, int, float] = GymDataset(env=env, observe_pixels=False)
    check_interaction_with_env(bob, obs_shape=(4,), action=1, reward_shape=(1,))


def test_wrap_cartpole_pixels():
    env = gym.make("CartPole-v0")
    bob: GymDataset[Tensor, int, float] = GymDataset(env=env, observe_pixels=True)
    check_interaction_with_env(bob, obs_shape=(400, 600, 3), action=1, reward_shape=(1,))


def test_cartpole_state():
    env: GymDataset[Tensor, int, float] = GymDataset("CartPole-v0", observe_pixels=False)
    check_interaction_with_env(env, obs_shape=(4,), action=1, reward_shape=(1,))


def test_cartpole_pixels():
    bob: GymDataset[Tensor, int, float] = GymDataset("CartPole-v0", observe_pixels=True)
    check_interaction_with_env(bob, obs_shape=(400, 600, 3), reward_shape=(1,))
