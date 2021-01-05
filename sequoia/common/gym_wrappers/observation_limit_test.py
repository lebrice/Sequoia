from functools import partial

import gym
import numpy as np
import pytest
from gym.vector import SyncVectorEnv

from sequoia.conftest import DummyEnvironment

from .batch_env import BatchedVectorEnv
from .env_dataset import EnvDataset
from .observation_limit import ObservationLimit


@pytest.mark.parametrize("env_name", ["CartPole-v0"])
def test_step_limit_with_single_env(env_name: str):
    """ Env should close when a given number of observations have been produced
    """
    env = gym.make(env_name)
    env = ObservationLimit(env, max_steps=5)
    env.seed(123)
    
    
    done = False
    # First episode.
    obs = env.reset()
    obs, reward, done, info = env.step(env.action_space.sample())
    obs, reward, done, info = env.step(env.action_space.sample())
    obs = env.reset()
    obs, reward, done, info = env.step(env.action_space.sample())
    assert env.is_closed
    
    with pytest.raises(gym.error.ClosedEnvironmentError):
        env.reset()

    with pytest.raises(gym.error.ClosedEnvironmentError):
        env.step(env.action_space.sample())


@pytest.mark.xfail(reason="TODO: Fix the bugs in the interaction between "
                          "EnvDataset and ObservationLimit.")
@pytest.mark.parametrize("env_name", ["CartPole-v0"])
def test_step_limit_with_single_env_dataset(env_name: str):
    env = gym.make(env_name)
    start = 0
    target = 10
    env = DummyEnvironment(start=start, target=target, max_value=10 * 2)
    env = EnvDataset(env)
    
    max_steps = 5
    
    env = ObservationLimit(env, max_steps=max_steps)
    env.seed(123)
    values = []
    for i, obs in zip(range(100), env):
        values.append(obs)
        _ = env.send(1)
    assert values == list(range(start, max_steps))
    
    assert env.is_closed
    
    with pytest.raises(gym.error.ClosedEnvironmentError):
        env.reset()

    with pytest.raises(gym.error.ClosedEnvironmentError):
        env.step(env.action_space.sample())
            
    with pytest.raises(gym.error.ClosedEnvironmentError):
        for i, _ in zip(range(5), env):
            assert False
            

@pytest.mark.parametrize("batch_size", [3, 5])
def test_step_limit_with_vectorized_env(batch_size):
    start = 0
    target = 10
    starting_values = [start for i in range(batch_size)]
    targets = [target for i in range(batch_size)]
    
    env = SyncVectorEnv([
        partial(DummyEnvironment, start=start, target=target, max_value=target * 2)
        for start, target in zip(starting_values, targets)
    ])
    env = ObservationLimit(env, max_steps = 3 * batch_size)
    
    obs = env.reset()
    obs, reward, done, info = env.step(env.action_space.sample())
    # obs, reward, done, info = env.step(env.action_space.sample())
    obs = env.reset()
    assert env.is_closed
    
    with pytest.raises(gym.error.ClosedEnvironmentError):
        env.reset()
    
    with pytest.raises(gym.error.ClosedEnvironmentError):
        _ = env.step(env.action_space.sample())


@pytest.mark.parametrize("batch_size", [3, 5])
def test_step_limit_with_vectorized_env_partial_final_batch(batch_size):
    """ In the case where the batch size isn't a multiple of the max
    observations, the env returns ceil(max_obs / batch_size) * batch_size
    observations in total.

    TODO: If we ever get to few-shot learning or something like that, we might
    have to care about this.
    """
    start = 0
    target = 10
    starting_values = [start for i in range(batch_size)]
    targets = [target for i in range(batch_size)]
    
    env = SyncVectorEnv([
        partial(DummyEnvironment, start=start, target=target, max_value=target * 2)
        for start, target in zip(starting_values, targets)
    ])
    env = ObservationLimit(env, max_steps = 3 * batch_size + 1)
    
    obs = env.reset()
    assert not env.is_closed
    
    obs, reward, done, info = env.step(env.action_space.sample())
    obs, reward, done, info = env.step(env.action_space.sample())
    assert not env.is_closed
    
    # obs, reward, done, info = env.step(env.action_space.sample())
    obs = env.reset()
    assert env.is_closed
    
    with pytest.raises(gym.error.ClosedEnvironmentError):
        env.reset()
    
    with pytest.raises(gym.error.ClosedEnvironmentError):
        _ = env.step(env.action_space.sample())
    