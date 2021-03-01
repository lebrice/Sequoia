""" TODO: Tests for this new 'multi-task Env' (v2)

"""
from typing import List

import gym
from gym.envs.classic_control import CartPoleEnv

from ..env_dataset import EnvDataset
from .change_after_each_episode import ChangeAfterEachEpisode
from .multi_task_env import NamedTuple


def test_basics():
    nb_tasks = 5
    envs = [CartPoleEnv() for _ in range(nb_tasks)]
    lengths = {
        # Add offset since length = 0 causes nans in obs.
        i: 0.2 + 0.1 * i
        for i in range(nb_tasks)
    }
    for i, env in enumerate(envs):
        env.unwrapped.length = lengths[i]

    env = ChangeAfterEachEpisode(envs)
    env.seed_all(123)

    episode_task_indices: List[int] = []
    episode_lengths: List[int] = []

    n_episodes = 5
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        episode_length = 0

        episode_task_index = obs[1]
        assert isinstance(episode_task_index, int)
        episode_task_indices.append(episode_task_index)

        assert env.length == lengths[episode_task_index]

        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            assert obs[1] == episode_task_index
            episode_length += 1

        episode_lengths.append(episode_length)

    assert len(set(episode_task_indices)) > 1


def test_iteration():
    """ Test that iterating through one of these MultiTaskEnvs works as expected. """
    nb_tasks = 5
    envs = [EnvDataset(CartPoleEnv()) for _ in range(nb_tasks)]
    lengths = {
        # Add offset since length = 0 causes nans in obs.
        i: 0.2 + 0.1 * i
        for i in range(nb_tasks)
    }
    for i, env in enumerate(envs):
        env.unwrapped.length = lengths[i]

    env = ChangeAfterEachEpisode(envs)
    env.seed(123)
    nb_episodes = 5

    task_ids: List[int] = []

    for episode in range(nb_episodes):
        task_index = None
        for i, obs in enumerate(env):
            assert isinstance(obs, NamedTuple)
            if i == 0:
                task_index = obs[1]
                task_ids.append(task_index)
            else:
                assert obs[1] == task_index

            action = env.action_space.sample()
            reward = env.send(action)

    assert len(set(task_ids)) > 1
