from typing import Callable
from gym.vector.vector_env import VectorEnv
from sequoia.common.episode_collector.update_strategy import redo_forward_pass_strategy
from .episode_collector import EpisodeCollector
from .policy import RandomPolicy
import gym
import pytest
from functools import partial


@pytest.mark.parametrize(
    "env_fn",
    [
        partial(gym.make, "CartPole-v0"),
        partial(gym.vector.make, "CartPole-v0", num_envs=10),
    ],
)
def test_episode_collector(env_fn):
    env = env_fn()

    episode_collector = EpisodeCollector(env, policy=RandomPolicy())

    episodes = []
    for i, episode in zip(range(10), episode_collector):
        assert i < 10
        episodes.append(episode)

    assert i == 9
    assert len(episodes) == 10


@pytest.mark.parametrize(
    "env_fn",
    [
        partial(gym.make, "CartPole-v0"),
        partial(gym.vector.make, "CartPole-v0", num_envs=10),
    ],
)
def test_max_episodes(env_fn):
    env: gym.Env = env_fn()

    episode_collector = EpisodeCollector(env, policy=RandomPolicy(), max_episodes=10)

    episodes = []
    for i, episode in enumerate(episode_collector):
        assert i < 10
        episodes.append(episode)

    assert i == 9
    assert len(episodes) == 10


@pytest.mark.parametrize(
    "env_fn",
    [
        partial(gym.make, "CartPole-v0"),
        partial(gym.vector.make, "CartPole-v0", num_envs=10, asynchronous=False),
    ],
)
def test_max_episodes(env_fn):
    env = env_fn()
    episode_collector = EpisodeCollector(env, policy=RandomPolicy(), max_episodes=10)
    ep = next(episode_collector)


@pytest.mark.parametrize(
    "env_fn",
    [
        partial(gym.vector.make, "CartPole-v0", num_envs=10, asynchronous=False),
        partial(gym.make, "CartPole-v0"),
        partial(gym.make, "Pendulum-v1"),
     ],
)
@pytest.mark.parametrize("update_interval", [1, 2, 3, 10])
def test_redo_forward_passes(env_fn: Callable[[], VectorEnv], update_interval: int):
    env: VectorEnv = env_fn()
    policy = RandomPolicy()
    
    max_episodes = 50
    episode_collector = EpisodeCollector(
        env,
        policy=policy,
        max_episodes=max_episodes,
        what_to_do_after_update=redo_forward_pass_strategy,
    )
    updates_so_far = 0
    episodes = []

    for i, episode in enumerate(episode_collector):
        # since we redo the forward pass for the portions that need it after each policy update, we
        # should always get episodes from the most recent policy.
        assert set(episode.model_versions) == {updates_so_far}, (episode.model_versions)
        episodes.append(episode)

        if i != 0 and i % update_interval == 0:
            # Pretend like we're updating the model.
            episode_collector.send(RandomPolicy())
            updates_so_far += 1

        # Check that the max episodes has been respected:
        assert i < max_episodes

    assert i == max_episodes - 1
    assert len(episodes) == max_episodes
