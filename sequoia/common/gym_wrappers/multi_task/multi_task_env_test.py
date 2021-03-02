""" TODO: Tests for this new 'multi-task Env' (v2)

"""
from typing import Callable
import gym
from gym.envs.classic_control import CartPoleEnv

from .multi_task_env import MultiTaskEnv, NamedTuple


def test_basics():
    nb_tasks = 5
    envs = [CartPoleEnv() for _ in range(nb_tasks)]
    for i, env in enumerate(envs):
        env.length = 0.2 + 0.1 * i  # Add offset since length = 0 causes nans in obs.

    env = MultiTaskEnv(envs)

    n_episodes_per_task = 3
    for task_index in range(nb_tasks):
        env.change_task(task_index)
        assert env.unwrapped.length == 0.2 + 0.1 * task_index

        for episode in range(n_episodes_per_task):
            obs = env.reset()
            done = False
            while not done:
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
                assert obs[1] == task_index


def test_basics_with_env_fns():
    nb_tasks = 5

    lengths = {
        # Add offset since length = 0 causes nans in obs.
        i: 0.2 + 0.1 * i
        for i in range(nb_tasks)
    }

    def _make_env_fn(i) -> Callable[..., CartPoleEnv]:
        def _env_fn():
            env = CartPoleEnv()
            env.length = lengths[i]
            return env

        return _env_fn

    env = MultiTaskEnv([_make_env_fn(i) for i in range(nb_tasks)])
    env.seed(123)

    n_episodes_per_task = 3
    for task_index in range(nb_tasks):
        env.change_task(task_index)
        assert env.unwrapped.length == 0.2 + 0.1 * task_index

        for episode in range(n_episodes_per_task):
            obs = env.reset()
            done = False
            while not done:
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
                assert obs[1] == task_index


def test_iteration():
    """ Test that iterating through one of these MultiTaskEnvs works as expected. """
    nb_tasks = 5
    from sequoia.common.gym_wrappers import EnvDataset

    envs = [EnvDataset(CartPoleEnv()) for _ in range(nb_tasks)]
    for i, env in enumerate(envs):
        env.unwrapped.length = (
            0.2 + 0.1 * i
        )  # Add offset since length = 0 causes nans in obs.

    env = MultiTaskEnv(envs)

    n_episodes_per_task = 3
    for task_index in range(nb_tasks):
        env.change_task(task_index)
        assert env.unwrapped.length == 0.2 + 0.1 * task_index

        for episode in range(n_episodes_per_task):
            for i, obs in enumerate(env):
                assert isinstance(obs, NamedTuple)
                assert obs[1] == task_index

                action = env.action_space.sample()

                reward = env.send(action)
