""" CL environments based on the mujoco envs.

NOTE: This is based on https://github.com/Breakend/gym-extensions
"""
# from sequoia.conftest import mujoco_required
# pytestmark = mujoco_required

import os
from pathlib import Path

import gym
from gym.envs import register
from gym.envs.mujoco import MujocoEnv

from .half_cheetah import ContinualHalfCheetahEnv, HalfCheetahEnv, HalfCheetahGravityEnv
from .hopper import ContinualHopperEnv, HopperEnv, HopperGravityEnv
from .modified_gravity import ModifiedGravityEnv
from .modified_size import ModifiedSizeEnv
from .walker2d import ContinualWalker2dEnv, Walker2dEnv, Walker2dGravityEnv

SOURCE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))


from typing import Dict, List, Type


def get_entry_point(Env: Type[gym.Env]) -> str:
    # TODO: Make sure this also works when Sequoia is installed in non-editable mode.
    return f"{Env.__module__}:{Env.__name__}"


# The list of mujoco envs which we explicitly have support for.
# TODO: Should probably use a Wrapper rather than a new base class (at least for the
# GravityEnv and the modifications that can be made to an already-instantiated env.
# NOTE: Using the same version tag as the

CURRENTLY_SUPPORTED_MUJOCO_ENVS: Dict[str, Type[MujocoEnv]] = {
    "ContinualHalfCheetah-v2": ContinualHalfCheetahEnv,
    "Hopper-v2": ContinualHopperEnv,
    "Walker2d-v2": ContinualWalker2dEnv,
}


# TODO: Register the 'continual' variants automatically by finding the entries in the
# registry that can be wrapped, and wrapping them.


gym.envs.register(
    id="ContinualHalfCheetah-v2",
    entry_point=get_entry_point(ContinualHalfCheetahEnv),
    max_episode_steps=1000,
    reward_threshold=4800.0,
)


gym.envs.register(
    id="ContinualHopper-v2",
    entry_point=get_entry_point(ContinualHopperEnv),
    max_episode_steps=1000,
    reward_threshold=4800.0,
)


gym.envs.register(
    id="ContinualWalker2d-v3",
    entry_point=get_entry_point(ContinualWalker2dEnv),
    max_episode_steps=1000,
    reward_threshold=4800.0,
)
