from typing import Tuple
import gym
from gym import Env, Space
from .spaces import EnvSpace, Boxes, Discretes
import numpy as np


def test_env_in_space():
    algo_space: Space[Env] = EnvSpace(
        observation_space=Boxes(shape=lambda shape: np.prod(shape) < 100),
        action_space=Discretes(n=lambda n: n <= 2),
    )

    cartpole = gym.make("CartPole-v1")
    assert cartpole in algo_space
    unsupported_env = gym.make("Pendulum-v1")
    assert unsupported_env not in algo_space


def test_sample_from_env_space():
    """IDEA: sample one of the environments that matches the specification!"""
    algo_space: Space[Env] = EnvSpace(
        observation_space=Boxes(shape=lambda shape: np.prod(shape) < 100),
        action_space=Discretes(n=lambda n: n <= 2),
    )
    eligible_env = algo_space.sample()
    assert isinstance(eligible_env, gym.Env)
    assert eligible_env in algo_space
