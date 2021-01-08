"""
Tests for the SemiSupervisedEnv wrapper.
"""
import itertools
from collections.abc import Sequence
from typing import Generator, Iterable, List, Optional, Tuple

import gym
import pytest

from .semi_supervised_env import SemiSupervisedEnv


def reward_iterator(env: gym.Env, total_rewards: int) -> Iterable[Optional[float]]:
    """ Iterator that yields the rewards from a (possibly batched) environment.
    Takes random actions. Used to simplify the testing here a bit.
    """
    env.reset()
    remaining = total_rewards
    while remaining > 0:
        _, reward, done, _ = env.step(env.action_space.sample())
        if reward is None or isinstance(reward, float):
            remaining -= 1
            yield reward
        else:
            rewards = list(reward)
            n_to_take = min(len(rewards), remaining)
            yield from reward[:n_to_take]
            remaining -= n_to_take
        if isinstance(done, bool):
            if done:
                env.reset()
        else:
            # Done is a list or array or something like that.
            if any(done):
                env.reset()
    env.close()


@pytest.mark.parametrize("env_name", ["CartPole-v0", "Pendulum-v0"])
def test_default_args_is_fully_labeled(env_name: str):
    env = gym.make(env_name)
    env = SemiSupervisedEnv(env)
    env.reset()
    
    total_rewards: int = 100
    none_reward_count: int = 0
    expected_none_reward_count: int = 0
    
    for reward in reward_iterator(env, total_rewards):
        if reward is None:
            none_reward_count += 1

    assert none_reward_count == expected_none_reward_count
    env.close()


@pytest.mark.parametrize("env_name", ["CartPole-v0", "Pendulum-v0"])
@pytest.mark.parametrize("labeled_fraction", [0., 0.5, 0.9, 1.0])
@pytest.mark.parametrize("seed", [123, 456, 789, 1543])
def test_percent_labeled(env_name: str, labeled_fraction: float, seed: int):
    env = gym.make(env_name)
    env = SemiSupervisedEnv(env, labeled_fraction=labeled_fraction)
    env.reset()
    env.seed(seed)

    total_rewards: int = 1000
    expected_none_reward_count = round(total_rewards * (1-labeled_fraction))

    none_reward_count: int = 0
    for reward in reward_iterator(env, total_rewards):
        if reward is None:
            none_reward_count += 1
    env.close()

    # pretty loose 20% interval, sounds reasonable for testing.
    lower_bound = 0.8 * expected_none_reward_count
    upper_bound = 1.2 * expected_none_reward_count
    assert 0 <= none_reward_count <= total_rewards
    assert lower_bound <= none_reward_count
    assert none_reward_count <= upper_bound


@pytest.mark.parametrize("env_name", ["CartPole-v0", "Pendulum-v0"])
@pytest.mark.parametrize("max_labeled_samples", [0, 100, 10000])
@pytest.mark.parametrize("seed", [123, 456, 789, 1543])
def test_max_labeled_samples(env_name: str, max_labeled_samples: int, seed: int):
    env = gym.make(env_name)
    # This will give the first 'max_labeled_samples' rewards, since the labeled
    # fraction 1 by default.
    env = SemiSupervisedEnv(env, max_labeled_samples=max_labeled_samples)
    env.reset()
    env.seed(seed)

    total_rewards: int = 1000
    expected_reward_count = min(total_rewards, max_labeled_samples)

    reward_count: int = 0
    for reward in reward_iterator(env, total_rewards):
        if reward is not None:
            reward_count += 1
    env.close()

    assert reward_count == expected_reward_count



@pytest.mark.parametrize("env_name", ["CartPole-v0", "Pendulum-v0"])
@pytest.mark.parametrize("max_labeled_samples", [0, 100, 10000])
@pytest.mark.parametrize("labeled_fraction", [0., 0.5, 1.0])
@pytest.mark.parametrize("seed", [123, 456, 789])
def test_max_labeled_samples_and_labeled_fraction(env_name: str,
                                                  max_labeled_samples: int,
                                                  labeled_fraction: float,
                                                  seed: int):
    env = gym.make(env_name)
    env = SemiSupervisedEnv(
        env,
        labeled_fraction=labeled_fraction,
        max_labeled_samples=max_labeled_samples
    )
    env.reset()
    env.seed(seed)

    total_rewards: int = 1000
    reward_count: int = 0
    for reward in reward_iterator(env, total_rewards):
        if reward is not None:
            reward_count += 1
    env.close()
    
    assert 0 <= reward_count <= total_rewards
    
    if max_labeled_samples > total_rewards:
        lower_bound = 0.8 * total_rewards * labeled_fraction
        upper_bound = 1.2 * total_rewards * labeled_fraction
    else:
        lower_bound = min(
            0.8 * total_rewards * labeled_fraction,
            max_labeled_samples
        )
        upper_bound = min(
            1.2 * total_rewards * labeled_fraction,
            max_labeled_samples
        )
    assert lower_bound <= reward_count
    assert reward_count <= upper_bound


@pytest.mark.parametrize("seed", [123])
@pytest.mark.parametrize("max_labeled_samples", [0, 100, 10000])
@pytest.mark.parametrize("labeled_fraction", [0., 0.5, 1.0])
@pytest.mark.parametrize("env_name", ["CartPole-v0"])
def test_max_labeled_samples_and_labeled_fraction_batched(env_name: str,
                                                          max_labeled_samples: int,
                                                          labeled_fraction: float,
                                                          seed: int):
    from functools import partial

    from .batch_env import AsyncVectorEnv
    batch_size: int = 4
    env = gym.make(env_name)
    env = AsyncVectorEnv([partial(gym.make, env_name) for i in range(batch_size)])
    env = SemiSupervisedEnv(
        env,
        labeled_fraction=labeled_fraction,
        max_labeled_samples=max_labeled_samples
    )
    env.reset()
    env.seed(seed)

    total_rewards: int = 1000
    reward_count: int = 0
    for reward in reward_iterator(env, total_rewards):
        if reward is not None:
            reward_count += 1
    env.close()
    
    assert 0 <= reward_count <= total_rewards
    
    if max_labeled_samples > total_rewards:
        lower_bound = 0.8 * total_rewards * labeled_fraction
        upper_bound = 1.2 * total_rewards * labeled_fraction
    else:
        lower_bound = min(
            0.8 * total_rewards * labeled_fraction,
            max_labeled_samples
        )
        upper_bound = min(
            1.2 * total_rewards * labeled_fraction,
            max_labeled_samples
        )
    assert lower_bound <= reward_count
    assert reward_count <= upper_bound

