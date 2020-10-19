import gym
import numpy as np

from .pixel_observation import PixelObservationWrapper


def test_passing_string_to_constructor():
    env = PixelObservationWrapper("CartPole-v0")
    assert env.observation_space.shape == (400, 600, 3)

def test_observation_space():
    env = PixelObservationWrapper(gym.make("CartPole-v0"))
    assert env.observation_space.shape == (400, 600, 3)

def test_reset_gives_pixels():
    with PixelObservationWrapper(gym.make("CartPole-v0")) as env:
        start_state = env.reset()
        assert start_state.shape == (400, 600, 3)
        assert start_state.dtype == np.uint8

def test_step_obs_is_pixels():
    with PixelObservationWrapper(gym.make("CartPole-v0")) as env:
        env.reset()
        obs, _, _, _ = env.step(env.action_space.sample())
        assert obs.shape == (400, 600, 3)
        assert obs.dtype == np.uint8

def test_state_attribute_is_pixels():
    with PixelObservationWrapper(gym.make("CartPole-v0")) as env:
        env.reset()
        assert env.state.shape == (400, 600, 3)
        assert env.state.dtype == np.uint8


def test_render_rgb_array():
    with PixelObservationWrapper(gym.make("CartPole-v0")) as env:
        window = env.viewer.window
        for i in range(50):
            obs, _, done, _ = env.step(env.action_space.sample())
            state = env.render(mode="rgb_array")
            assert state.shape == (400, 600, 3)
            assert state.dtype == np.uint8
            if done:
                env.reset()

def test_render_with_human_mode():
    with PixelObservationWrapper(gym.make("CartPole-v0")) as env:
        window = env.viewer.window
        for i in range(50):
            obs, _, done, _ = env.step(env.action_space.sample())
            env.render(mode="human")
            assert obs.shape == (400, 600, 3)
            if done:
                env.reset()
        assert env.viewer.window is window



def test_render_with_human_mode_with_env_dataset():
    from .env_dataset import EnvDataset
    with PixelObservationWrapper(gym.make("CartPole-v0")) as env:
        env = EnvDataset(env)
        window = env.viewer.window
        for i, batch in zip(range(500), env):
            obs, done, info = batch
            env.render(mode="human")
            assert obs.shape == (400, 600, 3)
            action = env.action_space.sample()
            rewards = env.send(action)
             
            if done:
                env.reset()
        assert env.viewer.window is window
