import copy
import json
from abc import ABC
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Dict, List, Type, Union

import gym
from gym.envs.registration import EnvSpec, registry

from sequoia.utils import get_logger

logger = get_logger(__file__)

# IDEA: Modify a copy of the gym registry?
# sequoia_registry = copy.deepcopy(registry)
sequoia_registry = registry

from .classic_control import PixelObservationWrapper, register_classic_control_variants
from .variant_spec import EnvVariantSpec

register_classic_control_variants(sequoia_registry)


ATARI_PY_INSTALLED = False
try:
    from ale_py.gym.environment import ALGymEnv

    AtariEnv = ALGymEnv

    ATARI_PY_INSTALLED = True
except (gym.error.DependencyNotInstalled, ImportError):

    class AtariEnv(gym.Env):
        pass


MONSTERKONG_INSTALLED = False
try:
    # Redirecting stdout because this import prints stuff.
    from .monsterkong import MetaMonsterKongEnv, register_monsterkong_variants

    register_monsterkong_variants(sequoia_registry)
    MONSTERKONG_INSTALLED = True

except ImportError:

    class MetaMonsterKongEnv(gym.Env):
        pass


MTENV_INSTALLED = False
mtenv_envs = []
try:
    from mtenv import MTEnv
    from mtenv.envs.registration import mtenv_registry

    mtenv_envs = [env_spec.id for env_spec in mtenv_registry.all()]
    MTENV_INSTALLED = True
except ImportError:
    # Create a 'dummy' class so we can safely use MTEnv in the type hints below.
    # Additionally, isinstance(some_env, MTEnv) will always fail when mtenv isn't
    # installed, which is good.
    class MTEnv(gym.Env):
        pass


MUJOCO_INSTALLED = False
try:
    import mujoco_py

    mj_path, _ = mujoco_py.utils.discover_mujoco()
    from gym.envs.mujoco import MujocoEnv

    from .mujoco import (
        ContinualHalfCheetahEnv,
        ContinualHalfCheetahV2Env,
        ContinualHalfCheetahV3Env,
        ContinualHopperEnv,
        ContinualHopperV2Env,
        ContinualHopperV3Env,
        ContinualWalker2dEnv,
        ContinualWalker2dV2Env,
        ContinualWalker2dV3Env,
        register_mujoco_variants,
    )

    register_mujoco_variants(env_registry=sequoia_registry)
    MUJOCO_INSTALLED = True
except (
    ImportError,
    AttributeError,
    ValueError,
    gym.error.DependencyNotInstalled,
) as exc:
    logger.debug(f"Couldn't import mujoco: ({exc})")
    # Create a 'dummy' class so we can safely use type hints everywhere.
    # Additionally, `isinstance(some_env, <this class>)`` will always fail when the
    # dependency isn't installed, which is good.
    class MujocoEnv(gym.Env):
        pass

    class ContinualHalfCheetahEnv(MujocoEnv):
        pass

    class ContinualHalfCheetahV2Env(MujocoEnv):
        pass

    class ContinualHalfCheetahV3Env(MujocoEnv):
        pass

    class ContinualHopperEnv(MujocoEnv):
        pass

    class ContinualHopperV2Env(MujocoEnv):
        pass

    class ContinualHopperV3Env(MujocoEnv):
        pass

    class ContinualWalker2dEnv(MujocoEnv):
        pass

    class ContinualWalker2dV2Env(MujocoEnv):
        pass

    class ContinualWalker2dV3Env(MujocoEnv):
        pass


METAWORLD_INSTALLED = False
metaworld_envs: List[Type[gym.Env]] = []

try:
    if not MUJOCO_INSTALLED:
        # Skip the stuff below, since metaworld requires mujoco anyway.
        raise ImportError

    import metaworld
    from metaworld import MetaWorldEnv

    # TODO: Use mujoco from metaworld? or from mujoco_py?
    from metaworld.envs.mujoco.mujoco_env import MujocoEnv as MetaWorldMujocoEnv
    from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv

    # from metaworld.envs.mujoco.mujoco_env import MujocoEnv

    METAWORLD_INSTALLED = True
    # metaworld_dir = getsourcefile(metaworld)
    # mujoco_dir = Path("~/.mujoco").expanduser()
    # TODO: Cache the names of the metaworld envs to a file, just so we don't take about
    # 10 seconds to import metaworld every time?
    # TODO: Make sure this also works on a cluster.
    # TODO: When updating metaworld, need to remove this file.
    envs_cache_file = Path("temp/metaworld_envs.json")
    envs_cache_file.parent.mkdir(exist_ok=True)
    all_metaworld_envs: Dict[str, List[str]] = {}

    if envs_cache_file.exists():
        with open(envs_cache_file, "r") as f:
            all_metaworld_envs = json.load(f)
    else:
        print(
            "Loading up the list of available envs from metaworld for the first time, "
            "this might take a while (usually ~10 seconds)."
        )

    if "ML10" not in all_metaworld_envs:
        ML10_envs = list(metaworld.ML10().train_classes.keys())
        all_metaworld_envs["ML10"] = ML10_envs

    with open(envs_cache_file, "w") as f:
        json.dump(all_metaworld_envs, f)

    metaworld_envs = sum([list(envs) for envs in all_metaworld_envs.values()], [])
except (ImportError, AttributeError, gym.error.DependencyNotInstalled) as e:
    logger.debug(f"Unable to import metaworld: {e}")
    # raise e


if not METAWORLD_INSTALLED:
    # Create a 'dummy' class so we can safely use MetaWorldEnv in the type hints below.
    # Additionally, isinstance(some_env, MetaWorldEnv) will always fail when metaworld
    # isn't installed, which is good.
    class MetaWorldEnv(gym.Env, ABC):
        pass

    class MetaWorldMujocoEnv(gym.Env, ABC):
        pass

    class SawyerXYZEnv(gym.Env, ABC):
        pass
