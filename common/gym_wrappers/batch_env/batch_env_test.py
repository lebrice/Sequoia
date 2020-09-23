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
        for i in range(5):
            print(f"Starting state: {state}")
            print(f"Action: {actions}")
            expected_state_change = np.array([impact_on_state[action] for action in actions])
            print(f"Expected change on state: {expected_state_change}")
            expected_state = state + expected_state_change
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

@pytest.mark.parametrize("batch_size", [1, 2, 10])
def test_batch_env_datasets(batch_size: int):
    with BatchEnv(env_factory=DummyEnvironment, batch_size=batch_size) as env:
        env = EnvDataset(env)
        x = env.reset()
        assert x.tolist() == [0] * batch_size

        state = np.arange(batch_size)
        env.setattr_foreach("i", state)
        assert env.i == state.tolist()
        
        # There is the issue that the _observation field is still at the
        # previous value ([0,0]) when we start iterating, because we modified
        # the env, bypassing the actual step and everything.
        # Here I solve it quite simply by I solve it by doing one step in the
        # env with a no-op action. We 
        # state = np.arange(batch_size)
        actions = [0] * batch_size
        reward = env.send(actions)
        assert reward.tolist() == np.abs(5 - state).tolist()

        print("Before loop")
        for i, (obs, dones, info) in zip(range(1), env):
            print(f"Step {i}: {obs}, {dones}, {info}")
            assert obs.tolist() == (state + i).tolist()

            actions = [1] * batch_size # increment the state.
            reward = env.send(actions)
            
            # Just for this dummy env, we can easily keep track of what the state
            # should become. 
            state = (obs + 1) % 10
            assert reward.tolist() == np.abs(5 - (obs+ 1)).tolist()


@pytest.mark.parametrize("batch_size", [1, 2, 5, 10])
def test_partial_reset(batch_size: int):
    with BatchEnv(env_factory=DummyEnvironment, batch_size=batch_size) as env:
        env.reset()
        env.seed([123] * batch_size)

        for i in range(2):
            obs, reward, done, info = env.step(np.arange(batch_size) + 1 % 3)

        indices = np.arange(batch_size)
        even_indices = indices[indices % 2 == 0]
        odd_indices = indices[indices % 2 != 0]
        reset_mask = np.zeros(batch_size, dtype=bool)
        reset_mask[even_indices] = True

        assert not all(obs[even_indices] == 0)
        obs = env.partial_reset(reset_mask)
        assert all(obs[i] == 0 for i in even_indices), obs
        assert all(obs[i] == None for i in odd_indices), obs



def test_batch_size_one_doesnt_flatten():
    batch_size = 1

    with BatchEnv(env_factory=DummyEnvironment, batch_size=batch_size) as env:
            env.reset()
            env.seed([123] * batch_size)