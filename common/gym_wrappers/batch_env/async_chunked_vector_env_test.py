from .async_chunked_vector_env import AsyncChunkedVectorEnv

from functools import partial
import pytest
import gym
import pytest


@pytest.mark.parametrize("batch_size", [1, 5, 10])
def test_spaces_have_right_shape(batch_size: int):
    env_fn = partial(gym.make, "CartPole-v0")
    env_fns = [env_fn for _ in range(batch_size)]

    env = AsyncChunkedVectorEnv(env_fns, n_workers=4)
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
