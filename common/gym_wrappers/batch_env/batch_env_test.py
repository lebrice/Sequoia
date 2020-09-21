from typing import Dict

import gym
import numpy as np
import pytest
import torch
from baselines.common.vec_env import VecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from torch.utils.data import DataLoader

from conftest import DummyEnvironment
from settings.active.active_dataloader import ActiveDataLoader

from ..env_dataset import EnvDataset
from ..multi_task_environment import MultiTaskEnvironment
from .batch_env import BatchEnv


def env_factory():
    env = DummyEnvironment()
    return env

@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_getattr(batch_size: int):
    """ Tests that getting an attribute on the BatchEnv gets it from each env.
    """
    with BatchEnv(env_factory=env_factory, batch_size=batch_size) as env:
        state = env.reset()
        assert state.tolist() == [0] * batch_size

        actions = [i % 3 for i in range(batch_size)]
        impact_on_state: Dict[int, int] = {
            0: +0,
            1: +1,
            2: -1,
        }

        start_state = state
        print(f"Starting state: {state}")
        print(f"Action: {actions}")
        expected_state_change = np.array([impact_on_state[action] for action in actions])
        print(f"Expected change on state: {expected_state_change}")
        expected_state = start_state + expected_state_change
        expected_state %= 10 # (since the values wrap around when negative.)
        
        state, reward, done, info = env.step(actions)
        assert state.tolist() == expected_state.tolist()
        # This should also be equivalent:
        assert env.i == expected_state.tolist()

@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_get_missing_attr_raises_error(batch_size: int):
    """ Tests that getting a missing attribute on the BatchEnv tries to get it
    from each remote environment, and if it fails, raises an AttributeError.
    """
    with BatchEnv(env_factory=env_factory, batch_size=batch_size) as env:
        state = env.reset()
        assert state.tolist() == [0] * batch_size

        with pytest.raises(AttributeError):
            print(env.blablabob)

@pytest.mark.xfail(
    reason="TODO: Doesn't work quite yet. When a worker raises an "
           "AttributeError, it can't keep working normally after. "
           "Maybe that makes sense though?"
)
@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_get_missing_attr_raises_and_doesnt_crash_workers(batch_size: int):
    """ Tests that getting a missing attribute on the BatchEnv tries to get it
    from each remote environment, and if it fails, raises an AttributeError.
    """
    with BatchEnv(env_factory=env_factory, batch_size=batch_size) as env:
        state = env.reset()
        assert state.tolist() == [0] * batch_size
        
        with pytest.raises(AttributeError):
            print(env.blablabob)

        # Try to get the 'i' attribute from each of the envs:
        assert env.i == [0] * batch_size

        for i in range(5):
            assert env.i == [i] * batch_size
            observation, reward, done, info = env.step([1] * batch_size)
            assert all(observation[~done] == i + 1)
            assert all(observation[done] == 0)
            for obs, done in zip(observation, done):
                if not done:
                    assert obs == i + 1
                else:
                    assert obs == 0


@pytest.mark.parametrize("batch_size", [1, 2, 5])        
def test_setattr(batch_size: int):
    """ Make sure that setting an attribute sets it correctly on all the
    environments.
    """
    with BatchEnv(env_factory=DummyEnvironment, batch_size=batch_size) as env:
        state = env.reset()
        assert state.tolist() == [0] * batch_size

        env.setattr("i", 3)
        assert env.i == [3] * batch_size


@pytest.mark.parametrize("batch_size", [1, 2, 5])        
def test_setattr_foreach(batch_size: int):
    """ Make sure that setting an attribute sets the corresponding value on each
    environment.
    """
    with BatchEnv(env_factory=DummyEnvironment, batch_size=batch_size) as env:
        state = env.reset()
        assert state.tolist() == [0] * batch_size

        env.setattr_foreach("i", np.arange(batch_size))
        assert env.i == np.arange(batch_size).tolist()


def test_batch_env_datasets():
    batch_size = 2
    with BatchEnv(env_factory=DummyEnvironment, batch_size=batch_size) as env:
        env = EnvDataset(env)
        x = env.reset().tolist()
        assert x == [0] * batch_size

        env.setattr_foreach("i", np.arange(batch_size))

        for i, (x, done, info) in zip(range(2), env):
            assert x.tolist() == [i, i+1]

            actions = [0] * batch_size
            reward = env.send(actions)
            assert reward.tolist() == np.abs(5 - x).tolist()
    

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
