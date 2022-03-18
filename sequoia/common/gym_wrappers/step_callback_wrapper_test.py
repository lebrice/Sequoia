from typing import Tuple

import gym

from .step_callback_wrapper import PeriodicCallback, StepCallback, StepCallbackWrapper

i: int = 0


def increment_i(step: int, env: gym.Env, step_results: Tuple):
    global i
    print(f"Incrementing i at step {step}: ({i} -> {i+1})")
    i += 1


def decrement_i(step: int, env: gym.Env, step_results: Tuple):
    global i
    print(f"Decrementing i at step {step}: ({i} -> {i-1})")
    i -= 1


def test_step_callback():
    callback = StepCallback(step=7, func=increment_i)
    env = StepCallbackWrapper(gym.make("CartPole-v0"), callbacks=[callback])
    env.reset()
    global i
    i = 0
    for step in range(10):
        obs, reward, done, info = env.step(env.action_space.sample())

        if step < 7:
            assert i == 0
        else:
            assert i == 1
        if done:
            env.reset()
    env.close()


def test_periodic_callback():
    global i
    i = 0
    inc_callback = PeriodicCallback(period=5, func=increment_i)
    dec_callback = PeriodicCallback(period=5, func=decrement_i, offset=2)
    env = StepCallbackWrapper(gym.make("CartPole-v0"), callbacks=[inc_callback, dec_callback])
    env.reset()

    def _next(env) -> int:
        obs, reward, done, info = env.step(env.action_space.sample())
        if done:
            env.reset()
        return i

    assert _next(env) == 1
    assert _next(env) == 1
    assert _next(env) == 0
    assert _next(env) == 0
    assert _next(env) == 0

    assert _next(env) == 1
    assert _next(env) == 1
    assert _next(env) == 0
    assert _next(env) == 0
    assert _next(env) == 0

    env.close()
