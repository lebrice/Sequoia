
from gym.envs.registration import register, spec, EnvRegistry, EnvSpec, registry
from gym.envs.classic_control import (
    CartPoleEnv,
    MountainCarEnv,
    Continuous_MountainCarEnv,
    PendulumEnv,
    AcrobotEnv,
)
from sequoia.common.gym_wrappers.pixel_observation import PixelObservationWrapper
from typing import Dict, Type
import gym


# TODO: Register the 'continual' variants automatically by finding the entries in the
# registry that can be wrapped, and wrapping them.

# def ContinualCartPole()


def get_entry_point(Env: Type[gym.Env]) -> str:
    return f"{Env.__module__}:{Env.__name__}"


# TODO: Register a custom 'Continual{}
# def ContinualCartPole()
env_spec: EnvSpec

classic_control_specs = {
    env_id: env_spec
    for env_id, env_spec in registry.env_specs.items()
    if any(
        env_id.startswith(name)
        for name in [
            "CartPole",
            "Pendulum",
            "Acrobot",
            "MountainCar",
            "ContinuousMountainCar",
        ]
    )
}
import copy

for env_id, env_spec in classic_control_specs.items():
    print(env_spec.entry_point == CartPoleEnv.__module__)
    new_env_id = f"Continual{env_id}"
    new_spec = copy.deepcopy(env_spec)
    
    register(new_env_id,
             entry_point=None,
             reward_threshold=None,
             nondeterministic=False,
             max_episode_steps=None,
             kwargs=None,
    )

assert False, classic_control_specs


gym.envs.register(
    id="ContinualHalfCheetah-v2",
    entry_point=get_entry_point(ContinualHalfCheetahEnv),
    max_episode_steps=1000,
    reward_threshold=4800.0,
)
