from io import StringIO
from contextlib import redirect_stdout
from gym import spaces
import numpy as np
from gym.envs.registration import EnvRegistry, registry, EnvSpec

# Avoid print statements from pygame package.
with redirect_stdout(StringIO()):
    from meta_monsterkong.make_env import MetaMonsterKongEnv, MkConfig
from .variant_spec import EnvVariantSpec


def observe_state(env: MetaMonsterKongEnv) -> MetaMonsterKongEnv:
    if not env.observe_state:
        env.unwrapped.observe_state = True
        env.unwrapped.observation_space = spaces.Box(0, 292, [402,], np.int16)
    return env


def register_monsterkong_variants(env_registry: EnvRegistry = registry) -> None:
    for env_id in ["MetaMonsterKong-v0", "MetaMonsterKong-v1"]:
        spec: EnvSpec = env_registry.spec(env_id)

        # Add an explicit 'State' variant of the envs.
        new_env_id = "State" + env_id
        new_spec = EnvVariantSpec.of(
            spec,
            new_id=new_env_id,
            new_max_episode_steps=500,
            new_kwargs={"observe_state": True},
        )
        if new_env_id not in env_registry.env_specs:
            env_registry.env_specs[new_env_id] = new_spec

        # Add an explicit 'Pixel' variant of the envs (even though by default we currently
        # always observe the state).
        new_env_id = "Pixel" + env_id
        new_spec = EnvVariantSpec.of(
            spec,
            new_id=new_env_id,
            new_max_episode_steps=500,
            new_kwargs={"observe_state": False},
        )
        if new_env_id not in env_registry.env_specs:
            env_registry.env_specs[new_env_id] = new_spec
