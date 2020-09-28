"""TODO: Add a Wrapper that makes an environment 'semi-supervised'.
"""
import random
from typing import Any, Optional, List, Iterable

import gym
from gym import Env, RewardWrapper, Wrapper
from gym.wrappers import TransformReward
from collections import abc
import sys
from utils.logging_utils import get_logger

logger = get_logger(__file__)


if sys.version_info < (3, 8):
    # Get it from a pypi backport if on python < 3.8
    from singledispatchmethod import singledispatchmethod
else:
    from functools import singledispatchmethod  # type: ignore


class SemiSupervisedEnv(RewardWrapper):
    def __init__(self,
                 env: gym.Env,
                 labeled_fraction: float = 1.0,
                 max_labeled_samples: Optional[int] = None):
        super().__init__(env)
        self.labeled_fraction = labeled_fraction
        self.max_labeled_samples = max_labeled_samples
        self.labeled_count: int = 0
        self.unlabeled_count: int = 0

    def seed(self, seed: int = None):
        random.seed(seed)
        super().seed(seed)

    @singledispatchmethod
    def reward(self, reward: Any) -> Any:
        logger.debug(f"Don't know how to make {reward} semi-supervised, returning as-is.")
        # NOTE: This only gets called if no appropriate dispatch method is found. 
        return reward

    @reward.register
    def reward_float(self, reward: float) -> Optional[float]:
        if random.random() <= self.labeled_fraction:
            if self.max_labeled_samples is None or self.labeled_count < self.max_labeled_samples:
                self.labeled_count += 1
                return reward
        # We either lost the coin toss, or already gave out the maximum number
        # of labeled samples we were allowed to give. Therefore we return None.
        self.unlabeled_count += 1
        return None

    @reward.register(abc.Iterable)
    def reward_batch(self, reward: Iterable[float]) -> List[Optional[float]]:
        return list(map(self.reward, reward))
