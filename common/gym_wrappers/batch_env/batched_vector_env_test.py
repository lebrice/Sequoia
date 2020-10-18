from functools import partial
from multiprocessing import cpu_count
from typing import Optional

import gym
import numpy as np
import pytest

from .batched_vector_env import BatchedVectorEnv

N_CPUS = cpu_count()


@pytest.mark.parametrize("batch_size", [1, 5, N_CPUS, 11, 24])
@pytest.mark.parametrize("n_workers", [1, 3, N_CPUS])
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
        assert len(info) == batch_size

    env.close()

from conftest import DummyEnvironment


@pytest.mark.parametrize("batch_size", [1, 2, 5, N_CPUS, 10, 24])
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


# @pytest.mark.xfail(
#     reason="TODO: Need to think a bit about the 'reset' behaviour of the "
#     " 'worker', currently it resets for us and gives us the new initial "
#     "observation when 'done'=True. "
# )
def test_done_reset_behaviour():
    batch_size = 10
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
