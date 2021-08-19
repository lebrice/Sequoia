from typing import Tuple

import gym

from .step_callback_wrapper import StepCallbackWrapper

i: int = 0

def increment_i(step: int, *args, **kwargs):
    global i
    print(f"Incrementing i at step {step}: ({i} -> {i+1})")
    i += 1

def decrement_i(step: int, *args, **kwargs):
    global i
    print(f"Decrementing i at step {step}: ({i} -> {i-1})")
    i -= 1


def test_step_callback():
    env = StepCallbackWrapper(gym.make("CartPole-v0"))
    env.add_step_callback(step=7, callback=increment_i)
    env.reset()
    global i
    i = 0
    for step in range(10):
        obs, reward, done, info = env.step(env.action_space.sample())

        if step < 7:
            assert i == 0
        else:
            assert i == 1, step
        if done:
            env.reset()
    env.close()

from torch.utils.data import IterableDataset
import gym
import numpy as np
from typing import Generator, Optional, Generic, TypeVar, Tuple, Union, Iterator

ObservationType = TypeVar("ObservationType")
ActionType = TypeVar("ActionType")
RewardType = TypeVar("RewardType")


class EnvDataset(
    gym.Wrapper,
    IterableDataset,
):
    def __init__(self, env: gym.Env):
        super().__init__(env=env)
        self.env = env
        self._iterator: Optional[Generator] = None

    def __iter__(self):
        if self._iterator:
            self._iterator.close()
        self._iterator = self.iterate(self.env)
        return self._iterator

    @staticmethod
    def iterate(env) -> Generator[Union[ObservationType, RewardType], ActionType, None]:
        """Iterator / generator for a gym.Env."""
        try:
            observations = env.reset()
            done = False
            steps = 0
            while not done:
                print(f"Steps: {steps}, done={done}")
                actions = yield observations
                if actions is None:
                    raise RuntimeError("Need to send an action after each observation.")
                observations, rewards, done, info = env.step(actions)
                steps += 1
                yield rewards
        except GeneratorExit:
            print("closing")

    def send(self, actions: ActionType) -> RewardType:
        return self._iterator.send(actions)


def test_step_callback_iteration():
    env = gym.make("CartPole-v0")
    from gym.wrappers import TimeLimit
    env = TimeLimit(env, max_episode_steps=10)
    env = EnvDataset(env)
    env = StepCallbackWrapper(env)
    env.add_step_callback(step=7, callback=increment_i)
    global i
    i = 0
    for step, obs in enumerate(env):
        print(step, i, obs)
        if step <= 7:
            assert i == 0
        else:
            assert i == 1
        if step >= 100:
            break
        rewards = env.send(env.action_space.sample())
        env.render()
    env.close()

    


def test_periodic_callback():
    global i
    i = 0
    env = gym.make("CartPole-v0")
    env = StepCallbackWrapper(env)
    env.add_periodic_callback(increment_i, period=5)
    env.add_periodic_callback(decrement_i, period=5, offset=2)
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
