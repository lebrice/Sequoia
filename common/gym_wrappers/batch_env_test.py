from typing import Dict

import gym
import numpy as np
import pytest
import torch
from baselines.common.vec_env import VecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from torch.utils.data import DataLoader

from common.gym_wrappers.env_dataset_test import DummyEnvironment
from settings.active.active_dataloader import ActiveDataLoader

from .batch_env import BatchEnv
from .env_dataset import EnvDataset
from .multi_task_environment import MultiTaskEnvironment


def test_batch_passive_datasets():
    i: int = 0
    def env_factory():
        nonlocal i
        def _env_factory():
            env = DummyEnvironment(i)
            return env
        i += 1
        return _env_factory
    batch_size = 2
    env = BatchEnv(env_factory=env_factory(), batch_size=batch_size)
    env = EnvDataset(env)
    for i, (obs, done, info) in zip(range(5), env):
        assert obs == np.arange(batch_size) + i
        rewards = env.send(np.ones(batch_size))
        assert rewards == [abs(5 - i) for _ in range(batch_size)]
    assert obs == np.ones(batch_size) * 5

def test_zip_active_datasets():
    i: int = 0
    def env_factory():
        nonlocal i
        env = DummyEnvironment(i)
        i += 1
        return env
    env = BatchEnv(env_factory=env_factory, batch_size=2)
    x = next(env)
    assert x == [0, 1]
    for i, x in enumerate(env):
        assert x == [i+1, i+2]
        if i == 3:
            break
    assert x == [4, 5]
    
    env.send([0, 2])
    x = next(env)
    assert x == [5, 8]

@pytest.mark.xfail(reason="Don't yet have multi-worker active dataloader working.")
@pytest.mark.parametrize("n_workers", [0, 1, 2, 4, 8, 24])
def test_zip_dataset_multiple_workers(n_workers):
    """
    TODO: Test that the BatchEnv actually works with multiple workers.
    """  


def make_env(task: Dict = None):
    def _make_env():
        env = gym.make("CartPole-v0")
        env = MultiTaskEnvironment(env)
        if task:
            env.current_task = task
        return env
    return _make_env


def test_subproc_vec_env():
    batch_size: int = 10
    env_fns = [
        make_env() for i in range(batch_size)   
    ]
    env = SubprocVecEnv(env_fns, spaces=None, context='spawn', in_series=1)
    env.reset()
    for i in range(1):
        actions = [env.action_space.sample() for i in range(batch_size)]
        obs, reward, done, info = env.step(actions)
        assert obs.shape == (batch_size, 4)
        env.render(mode="human")
    env.close()
