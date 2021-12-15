from .episode_collector import EpisodeCollector
from .policy import RandomPolicy
import gym
import pytest
from functools import partial


@pytest.mark.parametrize("env_fn", [
    partial(gym.make, "CartPole-v0"),
    partial(gym.vector.make, "CartPole-v0", num_envs=10),
])
def test_episode_collector(env_fn):
    env = env_fn()
    
    episode_collector = EpisodeCollector(env, policy=RandomPolicy())

    episodes = []
    for i, episode in zip(range(10), episode_collector):
        assert i < 10
        episodes.append(episode)

    assert i == 9
    assert len(episodes) == 10


@pytest.mark.parametrize("env_fn", [
    partial(gym.make, "CartPole-v0"),
    partial(gym.vector.make, "CartPole-v0", num_envs=10),
])
def test_max_episodes(env_fn):
    env: gym.Env = env_fn()

    episode_collector = EpisodeCollector(env, policy=RandomPolicy(), max_episodes=10)

    episodes = []
    for i, episode in enumerate(episode_collector):
        assert i < 10
        episodes.append(episode)

    assert i == 9
    assert len(episodes) == 10


@pytest.mark.parametrize("env_fn", [
    partial(gym.make, "CartPole-v0"),
    partial(gym.vector.make, "CartPole-v0", num_envs=10),
])
def test_max_episodes(env_fn):
    env = env_fn()
    episode_collector = EpisodeCollector(env, policy=RandomPolicy(), max_episodes=10)

    ep = next(episode_collector)
