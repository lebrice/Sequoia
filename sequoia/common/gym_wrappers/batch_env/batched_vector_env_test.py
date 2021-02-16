import time
from functools import partial
from multiprocessing import cpu_count
from typing import Callable, List, Optional
from gym import spaces
import torch
import gym
import numpy as np
import pytest

from .batched_vector_env import BatchedVectorEnv

from sequoia.conftest import DummyEnvironment, slow_param, slow, atari_py_required, param_requires_atari_py

from sequoia.common.gym_wrappers.multi_task_environment import MultiTaskEnvironment


@atari_py_required
@pytest.mark.parametrize("batch_size", [1, 5, slow_param(11), slow_param(24)])
@pytest.mark.parametrize("n_workers", [1, 3, None])
def test_space_with_tuple_observations(batch_size: int, n_workers: Optional[int]):
    def make_env():
        env = gym.make("Breakout-v0")
        env = MultiTaskEnvironment(env, add_task_id_to_obs=True, add_task_dict_to_info=True)
        return env
    
    env_fn = make_env
    env_fns = [env_fn for _ in range(batch_size)]
    from .async_vector_env import AsyncVectorEnv
    env = BatchedVectorEnv(env_fns, n_workers=n_workers)
    # env = AsyncVectorEnv(env_fns)
    env.seed(123)
    
    assert env.observation_space == spaces.Tuple([
        spaces.Box(0, 255, (batch_size, 210, 160, 3), np.uint8),
        spaces.MultiDiscrete(np.ones(batch_size)),
    ])

    assert env.single_observation_space == spaces.Tuple([
        spaces.Box(0, 255, (210, 160, 3), np.uint8),
        spaces.Discrete(1),
    ])
    
    obs = env.reset()
    assert obs[0].shape == env.observation_space[0].shape 
    assert obs[1].shape == env.observation_space[1].shape 
    assert obs in env.observation_space
    
    actions = env.action_space.sample()
    step_obs, rewards, done, info = env.step(actions)
    assert step_obs in env.observation_space
    
    assert len(rewards) == batch_size
    assert len(done) == batch_size
    assert all([isinstance(v, bool) for v in done.tolist()]), [type(v) for v in done]
    assert len(info) == batch_size


@pytest.mark.parametrize("batch_size", [1, 5, slow_param(11), slow_param(24)])
@pytest.mark.parametrize("n_workers", [1, 3, None])
def test_right_shapes(batch_size: int, n_workers: Optional[int]):
    env_fn = partial(gym.make, "CartPole-v0")
    env_fns = [env_fn for _ in range(batch_size)]

    env = BatchedVectorEnv(env_fns, n_workers=n_workers)
    env.seed(123)
    assert env.observation_space.shape == (batch_size, 4)
    assert len(env.action_space) == batch_size
    
    obs = env.reset()
    assert obs.shape == (batch_size, 4)

    for i in range(3):
        actions = env.action_space.sample()
        assert actions in env.action_space
        obs, rewards, done, info = env.step(actions)
        assert obs.shape == (batch_size, 4)
        assert len(rewards) == batch_size
        assert len(done) == batch_size
        assert all([isinstance(v, bool) for v in done.tolist()]), [type(v) for v in done]
        assert len(info) == batch_size

    env.close()


@pytest.mark.parametrize("batch_size", [1, 2, 5, slow_param(10), slow_param(24)])
def test_ordering_of_env_fns_preserved(batch_size):
    """ Test that the order of the env_fns is also reproduced in the order of
    the observations, and that the actions are sent to the right environments.
    """
    target = 50
    env_fns = [
        partial(DummyEnvironment, start=i, target=target, max_value=100)
        for i in range(batch_size)
    ]
    env = BatchedVectorEnv(env_fns, n_workers=4)
    env.seed(123)
    obs = env.reset()
    assert obs.tolist() == list(range(batch_size))

    obs, reward, done, info = env.step(np.zeros(batch_size))
    assert obs.tolist() == list(range(batch_size))
    # Increment only the 'counters' at even indices.
    actions = [
        int(i % 2 == 0) for i in range(batch_size)
    ]
    obs, reward, done, info = env.step(actions)
    even = np.arange(batch_size) % 2 == 0
    odd = np.arange(batch_size) % 2 == 1
    assert obs[even].tolist() == (np.arange(batch_size) + 1)[even].tolist()
    assert obs[odd].tolist() == np.arange(batch_size)[odd].tolist(), (obs, obs[odd], actions)
    assert reward.tolist() == (np.ones(batch_size) * target - obs).tolist()

    env.close()


@pytest.mark.xfail(
    reason="TODO: Removed the 'final_state' part of the PR on the gym repo, so "
    "maybe it would be better to get rid of all this `batch_env` folder and "
    "just use the fork as a submodule."
)
@pytest.mark.parametrize("batch_size", [10])
def test_done_reset_behaviour(batch_size: int):
    n_workers = 4
    target = batch_size
    starting_values = np.arange(batch_size)
    env_fns = [
        partial(DummyEnvironment, start=start_i, target=target, max_value=target * 2)
        for start_i in starting_values
    ]
    env = BatchedVectorEnv(env_fns, n_workers=n_workers)
    env.seed(123)
    obs = env.reset()
    assert obs.tolist() == list(range(batch_size))

    # Increment all the counters.
    obs, reward, done, info = env.step(np.ones(batch_size))
    # Only the last env (at position batch_size-1) should have 'done=True',
    # since it reached the 'target' value of batch_size + 1 
    last_index = batch_size - 1
    is_last = np.arange(batch_size) == batch_size - 1
    
    assert done[last_index]
    assert all(done == is_last)
    # The observation at the last index should be the new 'starting'
    # observation.
    assert obs[~done].tolist() == (np.arange(batch_size) + 1)[~done].tolist()
    assert obs[done].tolist() == starting_values[done].tolist()

    # TODO: This here wouldn't work with the `SyncVectorEnv` from gym.vector,
    # because it doesn't keep the final observation at all, it just overwrites
    # it. Would have been Nice for it to be kept in the 'info' dict at least..

    # The 'info' dict should have the final state as an observation.
    assert info[last_index]["final_state"] == target
    assert all("final_state" not in info_i for info_i in info[:last_index])
    env.close()


def test_render_rgb_array():
    batch_size = 4
    env = BatchedVectorEnv([
        partial(gym.make, "CartPole-v0") for i in range(batch_size)
    ])
    env.reset()
    obs = env.render(mode="rgb_array")
    assert obs.shape == (batch_size, 400, 600, 3)
    env.close()


def test_render_human():
    batch_size = 4
    env = BatchedVectorEnv([
        partial(gym.make, "CartPole-v0") for i in range(batch_size)
    ])
    env.reset()
    with env:
        for i in range(100):
            actions = env.action_space.sample()
            obs, reward, done, info = env.step(actions)
            env.render(mode="human")
            env.viewer.window


@pytest.mark.parametrize("env_name", ["CartPole-v0", "Pendulum-v0", param_requires_atari_py("Breakout-v0")])
def test_with_pixelobservationwrapper_before_batch(env_name: str):
    """ Test out what happens if we put the PixelObservationWrapper before the 
    batching, i.e. in each of the environments.
    """
    batch_size = 5
    n_steps = 100
    n_workers = None
    
    from ..pixel_observation import PixelObservationWrapper
    
    def make_env():
        return PixelObservationWrapper(gym.make(env_name))
    setup_time, time_per_step = benchmark(batch_size, n_workers, make_env)
    print(f"Setup time: {setup_time}, time_per_step: {time_per_step}")
    


@pytest.mark.parametrize("env_name", ["CartPole-v0", "Pendulum-v0"])
def test_with_pixelobservationwrapper_after_batch(env_name: str):
    """ Test out what happens if we put the PixelObservationWrapper *after* the 
    batching, i.e. wrapping the batched environment.
    """
    batch_size = 5
    n_steps = 100
    n_workers = None
    
    from ..pixel_observation import PixelObservationWrapper
    
    def make_env():
        return gym.make(env_name)
    setup_time, time_per_step = benchmark(
        batch_size,
        n_workers,
        make_env,
        wrappers=[PixelObservationWrapper]
    )
    print(f"Setup time: {setup_time}, time_per_step: {time_per_step}")
    



def benchmark(batch_size: int,
              n_workers: Optional[int],
              env_fn: Callable,
              wrappers: List[Callable]=None,
              n_steps: int = 100):
    batch_size = 32
    n_steps = 100
    n_workers = None
    
    start_time = time.time()
    env = BatchedVectorEnv([env_fn for i in range(batch_size)],
                           n_workers=n_workers)

    wrappers = wrappers or []
    for wrapper in wrappers:
        env = wrapper(env)
    
    setup_time = time.time() - start_time

    run_start = time.time()
    env.reset()
    with env:
        for i in range(n_steps):
            actions = env.action_space.sample()
            obs, reward, done, info = env.step(actions)
            # env.render(mode="human")
            
    time_per_step = (time.time() - run_start) / n_steps
    return setup_time, time_per_step

