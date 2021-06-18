from functools import partial
from operator import attrgetter
from typing import Tuple

import gym
import pytest
from sequoia.common.gym_wrappers import MultiTaskEnvironment
from gym import Env, Wrapper
from gym.envs.classic_control import CartPoleEnv
from gym import spaces
from sequoia.conftest import param_requires_atari_py
from .async_vector_env import AsyncVectorEnv

@pytest.fixture()
def allow_remote_getattr(monkeypatch):
    assert hasattr(AsyncVectorEnv, "allow_remote_getattr")
    monkeypatch.setattr(AsyncVectorEnv, "allow_remote_getattr", True)


@pytest.mark.parametrize("env_name, expected_obs_shape", [
    ("CartPole-v0", (4,)),    
    param_requires_atari_py("Breakout-v0", (210, 160, 3)),    
])
@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_spaces(batch_size: int, env_name: str, expected_obs_shape: Tuple[int, ...]):
    env_fns = [partial(gym.make, env_name) for _ in range(batch_size)]
    env: AsyncVectorEnv
    
    expected_obs_batch_shape = (batch_size, *expected_obs_shape)
        
    with AsyncVectorEnv(env_fns=env_fns) as env:
        assert isinstance(env.observation_space, spaces.Box)
        assert env.observation_space.shape == expected_obs_batch_shape
        
        assert isinstance(env.action_space, spaces.Tuple)
        assert len(env.action_space.spaces) == batch_size
        for space in env.action_space.spaces:
            assert isinstance(space, spaces.Discrete)

        
        reset_obs = env.reset()
        assert reset_obs.shape == expected_obs_batch_shape
        
        for i in range(5):
            obs, reward, done, info = env.step(env.action_space.sample())
            assert obs.shape == expected_obs_batch_shape
            assert reward.shape == (batch_size,)


@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_apply(batch_size: int):
    env_fns = [partial(gym.make, "CartPole-v0") for _ in range(batch_size)]
    env: AsyncVectorEnv
    with AsyncVectorEnv(env_fns=env_fns) as env:
        results = env.apply(attrgetter("length"))
        # Get the pole lengths.
        assert results == [0.5 for i in range(batch_size)]



@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_getitem_as_proxy(batch_size: int):
    env_fns = [partial(gym.make, "CartPole-v0") for _ in range(batch_size)]
    with AsyncVectorEnv(env_fns=env_fns) as env:
        # Set the pole length to 2.0 but only in the first environment.
        env[0].length = 2.0
        assert env[0].length == 2.0

        lengths = env[:].length
        # Get the pole lengths, check that the first env has a different value!
        assert lengths == [2.0 if i == 0 else 0.5 for i in range(batch_size)]


def test_getitem_with_slice():
    batch_size: int = 4
    env_fns = [partial(gym.make, "CartPole-v0") for _ in range(batch_size)]
    env: AsyncVectorEnv[CartPoleEnv]
    with AsyncVectorEnv(env_fns=env_fns) as env:
        env: AsyncVectorEnv[CartPoleEnv] = env
        # Set the pole length of the first 2 environments to 2.0.
        env[0:2].length = 2.0
        assert env[0].length == 2.0
        assert env[1].length == 2.0
        assert env[2].length == 0.5
        assert env[3].length == 0.5

        assert env[1:].length == [2.0, 0.5, 0.5]

import numpy as np


def test_getitem_with_mask():
    batch_size: int = 4
    env_fns = [partial(gym.make, "CartPole-v0") for _ in range(batch_size)]
    env: AsyncVectorEnv[CartPoleEnv]
    with AsyncVectorEnv(env_fns=env_fns) as env:
        env: AsyncVectorEnv[CartPoleEnv] = env
        # Set the pole length of the first 2 environments to 2.0.
        indices = np.arange(batch_size)
        mask = np.zeros(batch_size, dtype=bool)
        mask[indices % 2 == 0] = True
        env[mask].length = 2.0
        assert env[0].length == 2.0
        assert env[1].length == 0.5
        assert env[2].length == 2.0
        assert env[3].length == 0.5


@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_getattr_fails_by_default(batch_size: int):
    env_fns = [partial(gym.make, "CartPole-v0") for _ in range(batch_size)]
    env: AsyncVectorEnv[CartPoleEnv]
    with AsyncVectorEnv(env_fns=env_fns) as env:
        # Set the pole length to 2.0 but only in the first environment.
        env[0].length = 2.0

        # Since the env doesn't have the attribute, it will try to get it from the envs.
        with pytest.raises(AttributeError):
            lengths_without_slice = env.length
        
        lengths_with_slice = env[:].length
        
        # Get the pole lengths, check that the first env has a different value!
        assert lengths_with_slice == [2.0 if i == 0 else 0.5 for i in range(batch_size)]



@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_getattr_gets_it_from_envs(batch_size: int, allow_remote_getattr):
    env_fns = [partial(gym.make, "CartPole-v0") for _ in range(batch_size)]
    env: AsyncVectorEnv[CartPoleEnv]
    with AsyncVectorEnv(env_fns=env_fns) as env:
        # Set the pole length to 2.0 but only in the first environment.
        env[0].length = 2.0

        # Since the env doesn't have the attribute, it will try to get it from the envs.
        lengths_without_slice = env.length
        lengths_with_slice = env[:].length
        assert lengths_with_slice == lengths_without_slice
        
        # Get the pole lengths, check that the first env has a different value!
        assert lengths_without_slice == [2.0 if i == 0 else 0.5 for i in range(batch_size)]



def get_task_schedule(env: Env):
    return env.task_schedule

def env_factory():
    return MultiTaskEnvironment(gym.make("CartPole-v0"))

@pytest.mark.parametrize("batch_size", [2, 5])
def test_setattr_sets_attr_on_first_wrapper_with_attribute(batch_size: int):
    env_fns = [
        env_factory for _ in range(batch_size)
    ]
    with env_factory() as temp_env:
        default_task = temp_env.current_task

    env: AsyncVectorEnv[MultiTaskEnvironment]
    with AsyncVectorEnv(env_fns=env_fns) as env:
        # Set the pole length to 2.0 but only in the first environment.
        env[0].task_schedule = {0: dict(length=2.0)}
        
        env.reset()
        env.step(env.action_space.sample())

        current_tasks = env[:].current_task
        print(f"Current tasks: {current_tasks}")
        assert current_tasks[0]["length"] == 2.0
        assert current_tasks[1]["length"] == 0.5
        
        # assert False, current_tasks
        # assert current_tasks[1] == default_task

class MyCartPoleEnv(CartPoleEnv):
    def scale_length(self, coef: float) -> int:
        self.length *= coef
        return self.length

@pytest.mark.parametrize("batch_size", [4])
def test_batched_method_call(batch_size: int, allow_remote_getattr):
    env_fns = [MyCartPoleEnv for _ in range(batch_size)]
    env: AsyncVectorEnv[MyCartPoleEnv]
    with AsyncVectorEnv(env_fns=env_fns) as env:
        lengths = env.length
        assert lengths == [0.5 for i in range(batch_size)]

        new_lengths = env[:2].scale_length(3.0)

        assert new_lengths == [1.5, 1.5]
        lengths = env.length
        assert lengths == [1.5, 1.5] + [0.5 for i in range(2, batch_size)]
