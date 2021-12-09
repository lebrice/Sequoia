from .typed_gym import _Env, _VectorEnv, Observation, _Action, _Reward
import gym


def test_protocols_match_with_actual_implementations():
    assert isinstance(gym.make("CartPole-v0"), _Env)
    assert isinstance(gym.vector.make("CartPole-v0", num_envs=2, asynchronous=False), _VectorEnv)
