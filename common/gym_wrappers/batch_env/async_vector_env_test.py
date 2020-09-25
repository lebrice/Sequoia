from functools import partial
from operator import attrgetter

import gym
import pytest
from common.gym_wrappers import MultiTaskEnvironment
from gym import Env, Wrapper
from gym.envs.classic_control import CartPoleEnv

from .async_vector_env import AsyncVectorEnv


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
    env: AsyncVectorEnv[CartPoleEnv]
    with AsyncVectorEnv(env_fns=env_fns) as env:
        env: AsyncVectorEnv[CartPoleEnv] = env
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
def test_getattr_gets_it_from_envs(batch_size: int):
    env_fns = [partial(gym.make, "CartPole-v0") for _ in range(batch_size)]
    env: AsyncVectorEnv[CartPoleEnv]
    with AsyncVectorEnv(env_fns=env_fns) as env:
        env: AsyncVectorEnv[CartPoleEnv] = env
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
        env: AsyncVectorEnv[CartPoleEnv] = env
        # Set the pole length to 2.0 but only in the first environment.
        env[0].task_schedule = {0: dict(length=2.0)}
        
        env.reset()
        env.step(env.random_actions())

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
def test_batched_method_call(batch_size: int):
    env_fns = [MyCartPoleEnv for _ in range(batch_size)]
    env: AsyncVectorEnv[MyCartPoleEnv]
    with AsyncVectorEnv(env_fns=env_fns) as env:
        env: AsyncVectorEnv[MyCartPoleEnv] = env

        lengths = env.length
        assert lengths == [0.5 for i in range(batch_size)]

        new_lengths = env[:2].scale_length(3.0)

        assert new_lengths == [1.5, 1.5]
        lengths = env.length
        assert lengths == [1.5, 1.5] + [0.5 for i in range(2, batch_size)]



@pytest.mark.parametrize("batch_size", [4])
def test_batched_nested_attribute_call(batch_size: int):
    env_fns = [MyCartPoleEnv for _ in range(batch_size)]
    env: AsyncVectorEnv[MyCartPoleEnv]
    with AsyncVectorEnv(env_fns=env_fns) as env:
        env: AsyncVectorEnv[MyCartPoleEnv] = env

        lengths = env.length
        assert lengths == [0.5 for i in range(batch_size)]

        new_lengths = env[:2].scale_length(3.0)
        assert new_lengths == [1.5, 1.5]
        lengths = env.length
        assert lengths == [1.5, 1.5] + [0.5 for i in range(2, batch_size)]

