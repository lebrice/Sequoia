import gym
from typing import (
    TypeVar,
)
from typing import Protocol

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
