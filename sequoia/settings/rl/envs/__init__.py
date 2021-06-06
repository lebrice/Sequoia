from abc import ABC
from contextlib import redirect_stdout
from io import StringIO
from typing import List, Dict, Union, List, Type
import json
import gym
from pathlib import Path
import copy
from sequoia.utils import get_logger
from gym.envs.registration import registry

logger = get_logger(__file__)

# IDEA: Modify a copy of the gym registry?
# sequoia_registry = copy.deepcopy(registry)
sequoia_registry = registry

from .variant_spec import EnvVariantSpec
from .classic_control import PixelObservationWrapper, register_classic_control_variants
register_classic_control_variants(sequoia_registry)


ATARI_PY_INSTALLED = False
try:
    # from .atari import *
    from gym.envs.atari import AtariEnv
    ATARI_PY_INSTALLED = True
except gym.error.DependencyNotInstalled:
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
    from gym.envs.mujoco import MujocoEnv
    from .mujoco import *
    import mujoco_py
    mj_path, _ = mujoco_py.utils.discover_mujoco()
    MUJOCO_INSTALLED = True

except (ImportError, AttributeError, ValueError, gym.error.DependencyNotInstalled) as exc:
    logger.debug(f"Couldn't import mujoco: ({exc})")
    # Create a 'dummy' class so we can safely use type hints everywhere.
    # Additionally, `isinstance(some_env, <this class>)`` will always fail when the
    # dependency isn't installed, which is good.
    class MujocoEnv(gym.Env):
        pass
    class HalfCheetahEnv(MujocoEnv):
        pass
    class HopperEnv(MujocoEnv):
        pass
    class Walker2dEnv(MujocoEnv):
        pass
    class ContinualHalfCheetahEnv(HalfCheetahEnv):
        pass
    class ContinualHopperEnv(HopperEnv):
        pass
    class ContinualWalker2dEnv(Walker2dEnv):
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
except (ImportError, AttributeError, gym.error.DependencyNotInstalled):
    pass


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
