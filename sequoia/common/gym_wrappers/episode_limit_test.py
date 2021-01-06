from functools import partial

import gym
import numpy as np
import pytest
from gym.vector import SyncVectorEnv

from sequoia.conftest import DummyEnvironment

from .batch_env import BatchedVectorEnv
from .episode_limit import EpisodeLimit
from .env_dataset import EnvDataset


@pytest.mark.parametrize("env_name", ["CartPole-v0"])
def test_episode_limit_with_single_env(env_name: str):
    """ EpisodeLimit should close the env when a given number of episodes is
    reached.
    """
    env = gym.make(env_name)
    env = EpisodeLimit(env, max_episodes=3)
    env.seed(123)
    
    done = False
    # First episode.
    obs = env.reset()
    while not done:
        obs, reward, done, info = env.step(env.action_space.sample())
    
    # Second episode.
    obs = env.reset()
    while not done:
        obs, reward, done, info = env.step(env.action_space.sample())
    
    # Third episode.
    obs = env.reset()
    while not done:
        obs, reward, done, info = env.step(env.action_space.sample())
    
    with pytest.raises(gym.error.ClosedEnvironmentError):
        obs = env.reset()

    with pytest.raises(gym.error.ClosedEnvironmentError):
        _ = env.step(env.action_space.sample())


@pytest.mark.xfail(reason="TODO: Fix the bugs in the interaction between "
                          "EnvDataset and EpisodeLimit.")
@pytest.mark.parametrize("env_name", ["CartPole-v0"])
def test_episode_limit_with_single_env_dataset(env_name: str):
    """ EpisodeLimit should close the env when a given number of episodes is
    reached when iterating through the env.
    """
    env = gym.make(env_name)
    env = EpisodeLimit(env, max_episodes=2)
    env = EnvDataset(env)
    # TODO: The reverse ordering doesn't work: (EnvDataset(EpisodeLimit))
    # TODO: There's a warning that doing this steps even though done = True?
    env.seed(123)

    done = False
    # First episode.
    for obs in env:
        print("in loop:", env.episode_count())
        reward = env.send(env.action_space.sample())

    print("between loops", env.episode_count())
    # Second episode.
    for i, obs in enumerate(env):
        print("Second loop", env.episode_count())
        reward = env.send(env.action_space.sample())

    # Trying to start a third episode should fail:
    with pytest.raises(gym.error.ClosedEnvironmentError):
        env.reset()
        for obs in env:
            assert False


@pytest.mark.parametrize("batch_size", [3, 5])
def test_episode_limit_with_vectorized_env(batch_size):
    """ Test that when adding the EpisodeLimit wrapper on top of a vectorized
    environment, the episode limit is with respect to each individual env rather
    than the batched env.
    """ 
    starting_values = [0 for i in range(batch_size)]
    targets = [10 for i in range(batch_size)]
    
    env = SyncVectorEnv([
        partial(DummyEnvironment, start=start, target=target, max_value=10 * 2)
        for start, target in zip(starting_values, targets)
    ])
    env = EpisodeLimit(env, max_episodes=2 * batch_size)
    
    obs = env.reset()
    assert obs.tolist() == starting_values 
    print("reset obs: ", obs)
    for i in range(10):
        print(i, obs)
        actions = np.ones(batch_size)
        obs, reward, done, info = env.step(actions)
    # all episodes end at step 10
    assert all(done)
    
    # Because of how VectorEnvs work, the obs are the new 'reset' obs, rather
    # than the final obs in the episode.
    assert obs.tolist() == starting_values 
    
    assert obs.tolist() == starting_values 
    print("reset obs: ", obs)
    for i in range(10):
        print(i, obs)
        actions = np.ones(batch_size)
        obs, reward, done, info = env.step(actions)

    # all episodes end at step 10
    assert all(done)
    assert env.is_closed
    assert obs.tolist() == starting_values
    with pytest.raises(gym.error.ClosedEnvironmentError):
        actions = np.ones(batch_size)
        obs, reward, done, info = env.step(actions)


@pytest.mark.xfail(reason="TODO: Fix the bugs in the interaction between "
                          "EnvDataset and EpisodeLimit.")
@pytest.mark.parametrize("batch_size", [3, 5])
def test_episode_limit_with_vectorized_env_dataset(batch_size):
    """ Test that when adding the EpisodeLimit wrapper on top of a vectorized
    environment, the episode limit is with respect to each individual env rather
    than the batched env.
    """
    start = 0
    target = 10
    starting_values = [start for i in range(batch_size)]
    targets = [target for i in range(batch_size)]

    env = SyncVectorEnv([
        partial(DummyEnvironment, start=start, target=target, max_value=10 * 2)
        for start, target in zip(starting_values, targets)
    ])
    
    max_episodes = 2
    
    env = EpisodeLimit(env, max_episodes=max_episodes * batch_size)
    env = EnvDataset(env)

    for i, obs in enumerate(env):
        print(i, obs)
        actions = np.ones(batch_size)
        reward = env.send(actions)

    assert  i == max_episodes * target - 1

    with pytest.raises(gym.error.ClosedEnvironmentError):
        env.reset()

    with pytest.raises(gym.error.ClosedEnvironmentError):
        for i, obs in enumerate(env):
            print(i, obs)
            actions = np.ones(batch_size)
            reward = env.send(actions)
    
    # all episodes end at step 10




# @pytest.mark.xfail(reason=f"BUG in EnvDataset, it doesn't finish ")
@pytest.mark.parametrize("batch_size", [3, 5])
def test_reset_vectorenv_with_unfinished_episodes_raises_warning(batch_size):
    """ Test that when adding the EpisodeLimit wrapper on top of a vectorized
    environment, the episode limit is with respect to each individual env rather
    than the batched env.
    """
    start = 0
    target = 10
    starting_values = [start for i in range(batch_size)]
    targets = [target for i in range(batch_size)]

    env = SyncVectorEnv([
        partial(DummyEnvironment, start=start, target=target, max_value=10 * 2)
        for start, target in zip(starting_values, targets)
    ])
    env = EpisodeLimit(env, max_episodes=3 * batch_size)
    
    obs = env.reset()
    _ = env.step(env.action_space.sample())
    _ = env.step(env.action_space.sample())
    with pytest.warns(UserWarning) as record:
        env.reset()
