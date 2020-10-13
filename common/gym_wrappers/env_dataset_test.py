import gym
import pytest
from gym.spaces import Discrete

from conftest import DummyEnvironment

from .env_dataset import EnvDataset


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

from .env_dataset import random_policy


def test_iterating_with_policy():
    env = DummyEnvironment()
    env = EnvDataset(env, max_episodes=1)
    env.seed(123)

    actions = [0, 1, 1, 2, 1, 1, 1, 1, 0, 0, 0]
    expected_obs = [0, 0, 1, 2, 1, 2, 3, 4, 5]
    expected_rewards = [5, 4, 3, 4, 3, 2, 1, 0]
    expected_dones = [False, False, False, False, False, False, False, True]

    reset_obs = 0
    # obs = env.reset()
    # assert obs == reset_obs
    n_calls = 0
    
    def custom_policy(observations, action_space):
        nonlocal n_calls
        action = actions[n_calls]
        n_calls += 1
        return action
    
    env.set_policy(custom_policy)

    for i, batch in enumerate(env):
        print(f"Step {i}: batch: {batch}")
        (observation, action, next_observation), reward = batch
        assert observation == expected_obs[i]
        assert next_observation == expected_obs[i+1]
        assert action == actions[i]
        assert reward == expected_rewards[i]

        if i == len(actions):
            break


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
            action = 0
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
        env.reset()
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

