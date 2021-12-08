from .typed_gym import Env, VectorEnv, Observation, Action, Reward
import gym


def test_protocols_match_with_actual_implementations():
    assert isinstance(gym.make("CartPole-v0"), Env)
    assert isinstance(gym.vector.make("CartPole-v0", num_envs=2, asynchronous=False), VectorEnv)
