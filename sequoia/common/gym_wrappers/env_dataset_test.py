import gym
import numpy as np
import pytest
import torch
from gym import spaces
from gym.spaces import Discrete

from sequoia.conftest import DummyEnvironment
from sequoia.common.transforms import Transforms
from .transform_wrappers import TransformObservation
from .env_dataset import EnvDataset

from sequoia.settings.active.rl.make_env import make_batched_env

def test_step_normally_works_fine():
    env = DummyEnvironment()
    env = EnvDataset(env)
    env.seed(123)
    
    obs = env.reset()
    assert obs == 0

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


def test_iterating_with_send():
    env = DummyEnvironment(target=5)
    env = EnvDataset(env)
    env.seed(123)

    actions = [0, 1, 1, 2, 1, 1, 1, 1, 0, 0, 0]
    expected_obs = [0, 0, 1, 2, 1, 2, 3, 4, 5]
    expected_rewards = [5, 4, 3, 4, 3, 2, 1, 0]
    expected_dones = [False, False, False, False, False, False, False, True]

    reset_obs = 0
    # obs = env.reset()
    # assert obs == reset_obs
    n_calls = 0

    for i, observation in enumerate(env):
        print(f"Step {i}: batch: {observation}")
        assert observation == expected_obs[i]
        
        action = actions[i]
        reward = env.send(action)
        assert reward == expected_rewards[i]
    # TODO: The episode will end as soon as 'done' is encountered, which means
    # that we will never be given the 'final' observation. In this case, the
    # DummyEnvironment will set done=True when the state is state = target = 5 
    # in this case.
    assert observation == 4


def test_raise_error_when_missing_action():
    env = DummyEnvironment()
    with EnvDataset(env) as env:
        env.reset()
        env.seed(123)

        with pytest.raises(RuntimeError):
            for i, observation in zip(range(5), env):
                pass

def test_doesnt_raise_error_when_action_sent():
    env = DummyEnvironment()
    with EnvDataset(env) as env:
        env.reset()
        env.seed(123)
    
        for i, obs in zip(range(5), env):
            assert obs in env.observation_space
            reward = env.send(env.action_space.sample())


def test_max_episodes():
    max_episodes = 3
    env = EnvDataset(
        env=gym.make("CartPole-v0"),
        max_episodes=max_episodes,
    )
    env.seed(123)
    for episode in range(max_episodes):
        # This makes use of the fact that given this seed, the episode should only
        # last a set number of frames.
        for i, observation in enumerate(env):
            print(f"step {i} {observation}")
            action = 0
            reward = env.send(action)
            if i >= 20:
                assert False, "The episode should never be longer than about 10 steps!"
    
    with pytest.raises(gym.error.ClosedEnvironmentError):
        for i, observation in enumerate(env):
            print(f"step {i} {observation}")
            env.send(env.action_space.sample())


def test_max_steps():
    epochs = 3
    max_steps = 5
    env = EnvDataset(
        env=gym.make("CartPole-v0"),
        max_steps=max_steps,
    )
    all_rewards = []
    all_observations = []
    with env:
        # TODO: Should we could what is given back by 'reset' as an observation?
        all_observations.append(env.reset())
        
        for i, batch in enumerate(env):
            assert i < max_steps, f"Max steps should have been respected: {i}"
            rewards = env.send(env.action_space.sample())
            all_rewards.append(rewards)
        assert len(all_rewards) == max_steps
        env.reset()

        with pytest.raises(gym.error.ClosedEnvironmentError):
            for i in range(10):
                print(i)
                observation = next(env)
                rewards = env.send(env.action_space.sample())
                all_rewards.append(rewards)
    
    assert len(all_rewards) == max_steps


def test_max_steps_per_episode():
    n_episodes = 4
    max_steps_per_episode = 5
    env = EnvDataset(
        env=gym.make("CartPole-v0"),
        max_steps_per_episode=max_steps_per_episode,
    )
    all_observations = []
    with env:
        for episode in range(n_episodes):
            env.reset()
            for i, batch in enumerate(env):
                assert i < max_steps_per_episode, f"Max steps per episode should have been respected: {i}"
                rewards = env.send(env.action_space.sample())
            assert i == max_steps_per_episode - 1


@pytest.mark.parametrize("env_name", ["CartPole-v0"])
@pytest.mark.parametrize("batch_size", [1, 2, 5, 10])
def test_not_setting_max_steps_per_episode_with_vector_env_raises_warning(env_name: str, batch_size: int):
    from gym.vector import SyncVectorEnv
    from functools import partial

    env = SyncVectorEnv([partial(gym.make, env_name) for i in range(batch_size)])
    with pytest.warns(UserWarning):
        dataset = EnvDataset(env)
    
    env.close()


class DummyWrapper(TransformObservation):
    def __init__(self, env, f=None):
        f = [torch.from_numpy, Transforms.channels_first, lambda t: t.numpy()]
        super().__init__(env, f=f)

    def observation(self, observation):
        print(f"Modifying an observation with shape {observation.shape}")
        result = super().observation(observation)
        print(f"Result shape: {result.shape}, max: {result.max()}")
        return result


def test_observation_wrapper_applies_to_yielded_objects():
    """ Test that when an TransformObservation wrapper (or any wrapper that
    changes the Observations) is applied on the env, the observations that are
    yielded by the GymDataLoader are also transformed, in the same way as those
    returned by step() or reset().
    """
    env_name = "Breakout-v0"
    batch_size = 10
    num_workers = 4
    max_steps_per_episode = 100
    wrapper = DummyWrapper

    vector_env = make_batched_env(env_name, batch_size=batch_size, num_workers=num_workers)
    env = EnvDataset(vector_env, max_steps_per_episode=max_steps_per_episode)
    
    assert env.observation_space == spaces.Box(0, 255, (10, 210, 160, 3), np.uint8) 

    # env = TransformObservation(env, f=[torch.from_numpy, Transforms.channels_first, lambda t: t.numpy()])
    env = DummyWrapper(env)
    assert env.observation_space == spaces.Box(0, 255, (10, 3, 210, 160), np.uint8)

    # env = DummyWrapper(env)
    # assert env.observation_space == spaces.Box(0, 255 // 2, (10, 210, 160, 3), np.uint8) 
    
    print("Before reset")
    reset_obs = env.reset()
    assert reset_obs in env.observation_space

    print("Before step")
    step_obs, _, _, _ = env.step(env.action_space.sample())
    assert step_obs in env.observation_space

    # We need to send an action before we can do this.
    action = env.action_space.sample()
    print(f"Before send")
    reward = env.send(action)
    
    print("Before __next__")
    next_obs = next(env)
    assert next_obs in env.observation_space
    
    print(f"Before iterating")
    # TODO: This still doesn't call the right .observation() method!
    
    for i, iter_obs in zip(range(3), env):
        assert iter_obs.shape == env.observation_space.shape
        assert iter_obs in env.observation_space

        action = env.action_space.sample()
        reward = env.send(action)

    env.close()


def test_iteration_with_more_than_one_wrapper():
    """ Same as above, but with more than one wrapper applied on top of the
    EnvDataset.
    """
    env_name = "Breakout-v0"
    batch_size = 10
    num_workers = 4
    max_steps_per_episode = 100
    wrapper = DummyWrapper
    
    vector_env = make_batched_env(env_name, batch_size=batch_size, num_workers=num_workers)
    env = EnvDataset(vector_env, max_steps_per_episode=max_steps_per_episode)

    assert env.observation_space == spaces.Box(0, 255, (10, 210, 160, 3), np.uint8) 

    env = DummyWrapper(env)
    assert env.observation_space == spaces.Box(0, 255, (10, 3, 210, 160), np.uint8)

    env = TransformObservation(env, f=[Transforms.to_tensor, Transforms.resize_64x64])
    assert env.observation_space == spaces.Box(0, 1.0, (10, 3, 64, 64), np.float32)
    # env = DummyWrapper(env)
    # assert env.observation_space == spaces.Box(0, 255 // 2, (10, 210, 160, 3), np.uint8) 

    print("Before reset")
    reset_obs = env.reset().numpy()
    assert reset_obs in env.observation_space

    print("Before step")
    step_obs, _, _, _ = env.step(env.action_space.sample())
    assert step_obs.numpy() in env.observation_space

    # We need to send an action before we can do this.
    action = env.action_space.sample()
    print(f"Before send")
    reward = env.send(action)
    
    print("Before __next__")
    next_obs = next(env).numpy()
    assert next_obs in env.observation_space
    
    print(f"Before iterating")
    # TODO: This still doesn't call the right .observation() method!

    for i, iter_obs in zip(range(3), env):
        assert iter_obs.shape == env.observation_space.shape
        assert iter_obs.numpy() in env.observation_space

        action = env.action_space.sample()
        reward = env.send(action)

    env.close()
