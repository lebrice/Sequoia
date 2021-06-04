""" Wrappers to use in order to force observing the state vs the pixels.
"""
from functools import singledispatch

import gym
from gym.envs.classic_control import (
    AcrobotEnv,
    CartPoleEnv,
    Continuous_MountainCarEnv,
    MountainCarEnv,
    PendulumEnv,
)
from sequoia.common.gym_wrappers.pixel_observation import PixelObservationWrapper


@singledispatch
def observe_pixels(env: gym.Env) -> gym.Env:
    raise NotImplementedError(f"Don't know how to force pixel observations for {env}")


@observe_pixels.register(CartPoleEnv)
@observe_pixels.register(MountainCarEnv)
@observe_pixels.register(PendulumEnv)
@observe_pixels.register(AcrobotEnv)
@observe_pixels.register(Continuous_MountainCarEnv)
def observe_pixels_classic_control_env(env: gym.Env) -> gym.Env:
    return PixelObservationWrapper(env)

@observe_pixels.register
def observe_pixels_for_wrapper(env: gym.Wrapper) -> gym.Wrapper:
    # TODO: TimeLimit<CartPoleEnv> should also work the same way!
    return observe_pixels.dispatch(type(env.unwrapped))(env)

@singledispatch
def observe_state(env: gym.Env) -> gym.Env:
    raise NotImplementedError(f"Don't know how to force state observations for {env}")

@observe_state.register
def observe_state_for_wrapper(env: gym.Wrapper) -> gym.Wrapper:
    # TODO: TimeLimit<CartPoleEnv> should also work the same way!
    return observe_state.dispatch(type(env.unwrapped))(env)
