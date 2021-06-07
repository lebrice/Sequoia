""" CL environments based on the mujoco envs.

NOTE: This is based on https://github.com/Breakend/gym-extensions
"""
# from sequoia.conftest import mujoco_required
# pytestmark = mujoco_required

import os
from pathlib import Path
from typing import Callable, Union

import gym
from gym.envs import register
from gym.envs.mujoco import MujocoEnv
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv
from gym.envs.registration import EnvRegistry, EnvSpec, load, registry
from sequoia.utils.logging_utils import get_logger

from ..variant_spec import EnvVariantSpec
from .half_cheetah import (
    ContinualHalfCheetahV2Env,
    ContinualHalfCheetahV3Env,
    HalfCheetahV2Env,
    HalfCheetahV3Env,
)
from .hopper import ContinualHopperEnv, HopperEnv
from .modified_gravity import ModifiedGravityEnv
from .modified_size import ModifiedSizeEnv
from .walker2d import (
    ContinualWalker2dV2Env,
    ContinualWalker2dV3Env,
    Walker2dV2Env,
    Walker2dV3Env,
)


logger = get_logger(__file__)

# NOTE: Prefer the 'V3' variants
# HalfCheetahEnv = HalfCheetahV3Env
# Walker2dEnv = Walker2dV3Env
ContinualHalfCheetahEnv = ContinualHalfCheetahV3Env
ContinualWalker2dEnv = ContinualWalker2dV3Env

SOURCE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

__all__ = [
    "ContinualHalfCheetahEnv",
    "ContinualHalfCheetahV2Env",
    "ContinualHalfCheetahV3Env",
    "ContinualHopperEnv",
    "ContinualWalker2dEnv",
    "ContinualWalker2dV3Env",
    "ModifiedGravityEnv",
    "ModifiedSizeEnv",
    "MujocoEnv",
]

from typing import Dict, List, Type


def get_entry_point(Env: Type[gym.Env]) -> str:
    # TODO: Make sure this also works when Sequoia is installed in non-editable mode.
    return f"{Env.__module__}:{Env.__name__}"


# The list of mujoco envs which we explicitly have support for.
# TODO: Should probably use a Wrapper rather than a new base class (at least for the
# GravityEnv and the modifications that can be made to an already-instantiated env.
# NOTE: Using the same version tag as the

CURRENTLY_SUPPORTED_MUJOCO_ENVS: Dict[str, Type[MujocoEnv]] = {
    "HalfCheetah-v2": ContinualHalfCheetahEnv,
    "HalfCheetah-v3": ContinualHalfCheetahV3Env,
    "Hopper-v2": ContinualHopperEnv,
    "Walker2d-v2": ContinualWalker2dV2Env,
    "Walker2d-v3": ContinualWalker2dV3Env,
}


# TODO: Register the 'continual' variants automatically by finding the entries in the
# registry that can be wrapped, and wrapping them.


# IDEA: Actually swap out the entries for these envs, rather than overwrite them?


def register_mujoco_variants(env_registry: EnvRegistry = registry) -> None:
    """ Adds pixel variants for the classic-control envs to the given registry in-place.
    """
    original_mujoco_env_specs: Dict[str, EnvSpec] = {
        spec.id: spec
        for env_id, spec in env_registry.env_specs.items()
        if isinstance(spec.entry_point, str)
        and spec.entry_point.startswith("gym.envs.mujoco")
    }
    # TODO: Add broader support for mujoco envs
    new_entry_points: Dict[
        str, Union[str, Callable[..., gym.Env]]
    ] = CURRENTLY_SUPPORTED_MUJOCO_ENVS
    supported_mujoco_env_specs = {
        env_id: spec
        for env_id, spec in original_mujoco_env_specs.items()
        if env_id in new_entry_points
    }
    for env_id, env_spec in supported_mujoco_env_specs.items():
        # TODO: Use the same ID, or a different one?
        # new_id = "Continual" + env_id
        new_id = env_id

        if new_id not in env_registry.env_specs or new_id == env_id:
            new_spec = EnvVariantSpec.of(
                original=env_spec,
                new_id=new_id,
                new_entry_point=new_entry_points[env_id],
            )
            env_registry.env_specs[new_id] = new_spec
            logger.debug(
                f"Registering MuJoCO Environment variant of {env_id} at id {new_id}."
            )


# Replace the entry-point for these mujoco envs.
# IMPORTANT: This doesn't change anything about the envs, apart from making it possible
# to explicitly change the gravity or mass etc if you want.
# TODO: Should probably still only modify a custom/copied registry, so that importing
# Sequoia doesn't modify the gym registry when Sequoia isn't being used explicitly.
# registry.env_specs["HalfCheetah-v2"].entry_point = ContinualHalfCheetahV2Env
# registry.env_specs["HalfCheetah-v3"].entry_point = ContinualHalfCheetahV3Env
# registry.env_specs["Hopper-v2"].entry_point = ContinualHopperEnv
# registry.env_specs["Walker2d-v2"].entry_point = ContinualWalker2dEnv

# EnvSpec(
#     "HalfCheetah-v2",
#     entry_point=get_entry_point(Continu),
#     reward_threshold=None,
#     nondeterministic=False,
#     max_episode_steps=None,
#     kwargs=None,
# )


# gym.envs.register(
#     id="ContinualHalfCheetah-v2",
#     entry_point=get_entry_point(ContinualHalfCheetahV2Env),
#     max_episode_steps=1000,
#     reward_threshold=4800.0,
# )

# gym.envs.register(
#     id="ContinualHalfCheetah-v3",
#     entry_point=get_entry_point(ContinualHalfCheetahV3Env),
#     max_episode_steps=1000,
#     reward_threshold=4800.0,
# )

# gym.envs.register(
#     id="ContinualHopper-v2",
#     entry_point=get_entry_point(ContinualHopperEnv),
#     max_episode_steps=1000,
#     reward_threshold=4800.0,
# )

# gym.envs.register(
#     id="ContinualWalker2d-v3",
#     entry_point=get_entry_point(ContinualWalker2dEnv),
#     max_episode_steps=1000,
#     reward_threshold=4800.0,
# )
