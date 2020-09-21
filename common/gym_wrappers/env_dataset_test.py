import gym
import pytest
from gym.spaces import Discrete

from conftest import DummyEnvironment

from .env_dataset import EnvDataset


def test_step_normally_works_fine():
    env = DummyEnvironment()
    env = EnvDataset(env)
    env.reset()
    env.seed(123)

    obs, reward, done, info = env.step(0)
    assert (obs, reward, done, info) == (0, 5, False, {})
    obs, reward, done, info = env.step(1)
    assert (obs, reward, done, info) == (1, 4, False, {})
    obs, reward, done, info = env.step(1)
    assert (obs, reward, done, info) == (2, 3, False, {})
    obs, reward, done, info = env.step(2)
    assert (obs, reward, done, info) == (1, 4, False, {})
    obs, reward, done, info = env.step(1)
    assert (obs, reward, done, info) == (2, 3, False, {})
    obs, reward, done, info = env.step(1)
    assert (obs, reward, done, info) == (3, 2, False, {})
    obs, reward, done, info = env.step(1)
    assert (obs, reward, done, info) == (4, 1, False, {})
    
    obs, reward, done, info = env.step(1)
    assert (obs, reward, done, info) == (5, 0, True, {})

    env.reset()
    obs, reward, done, info = env.step(0)
    assert (obs, reward, done, info) == (0, 5, False, {})

def test_raise_error_when_missing_action():
    env = DummyEnvironment()
    with EnvDataset(env) as env:
        env.reset()
        env.seed(123)
        iterator = iter(env)
        obs, done, info = next(iterator)
        assert obs == 0
        assert done is False
        with pytest.raises(RuntimeError):
            obs, done, info = next(iterator)
    # env = GymDataLoader(
    #     "CartPole-v0",
    #     batch_size=10,
    #     observe_pixels=False,
    #     random_actions_when_missing=False,
    # )
    # env.reset()
    # with pytest.raises(RuntimeError):
    #     for obs_batch in take(env, 5):
    #         # raises an error after the first iteration, as it didn't receive an action. 
    #         pass
