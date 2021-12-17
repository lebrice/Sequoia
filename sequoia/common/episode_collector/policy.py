import gym
from typing import (
    Optional,
    TypeVar,
)
from typing import Protocol

from numpy.random.mtrand import RandomState

Observation = TypeVar("Observation")
Observation_co = TypeVar("Observation_co", covariant=True)
Action = TypeVar("Action")
Reward = TypeVar("Reward")
Reward_co = TypeVar("Reward_co", covariant=True)
T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")

from sequoia.common.spaces.space import Space  # use the generic version of Space.


class Policy(Protocol[Observation, Action]):  # type: ignore
    def __call__(self, observation: Observation, action_space: Space[Action]) -> Action:
        raise NotImplementedError


class RandomPolicy(Policy[Observation, Action]):
    def __call__(self, observation: Observation, action_space: Space[Action]) -> Action:
        # todo: should we be garanteed to always have this be called with full batches?
        # i.e., constant batch size?
        return action_space.sample()


from torch import nn
import numpy as np


class EpsilonGreedyPolicy(nn.Module, Policy[Observation, Action]):
    def __init__(
        self, base_policy: Policy[Observation, Action], epsilon: float, seed: int = None
    ) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.base_policy = base_policy
        self.rng = np.random.RandomState(seed)

    def seed(self, seed: Optional[int]) -> None:
        self.rng = np.random.RandomState(seed)
    
    def __call__(self, observation: Observation, action_space: Space[Action]) -> Action:
        # Select a random action with probability epsilon.
        if self.rng.rand() < self.epsilon:
            return action_space.sample()
        return self.base_policy(observation, action_space=action_space)
