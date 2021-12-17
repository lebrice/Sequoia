from typing import Tuple
import gym
from .experience_replay import ExperienceReplayLoader
from gym import spaces
from sequoia.common.typed_gym import _Env, _Space


class SimpleEnv(gym.Env, _Env[int, int, float]):
    def __init__(self, target: int = 500, start_state: int = 0) -> None:
        super().__init__()
        self.observation_space = spaces.Discrete(1000)
        self.action_space = spaces.Discrete(2)
        self.target = target
        self.start_state = start_state
        self.state = start_state

    def reset(self) -> int:
        self.state = self.start_state
        return self.state

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        assert action in self.action_space
        self.state += action
        return self.state, -abs(self.state - self.target), self.state == self.target, {}


def test_experience_replay_simple():

    env = SimpleEnv()

    seed = 123
    # env.seed(seed)
    # env.observation_space.seed(seed)
    # env.action_space.seed(seed)
    # import numpy as np

    # np.random.seed(seed)
    # import random

    # random.seed(seed)
    # import torch
    
    # torch.random.manual_seed(seed)

    def policy(obs: int, action_space: _Space[int]) -> int:
        return action_space.sample()
        return 1
        # assert False, (obs, action_space)
        # return 0

    loader = ExperienceReplayLoader(
        env, batch_size=10, buffer_size=100, max_episodes=10, policy=policy, seed=seed,
    )
    # assert False, loader

    for i, transition_batch in enumerate(loader):
        assert False, transition_batch
    assert False, i
