""" Registers variants of the classic-control envs that are used by sequoia. """
# TODO: Add Pixel???-v? variants for the classic-control envs.
from typing import Dict

from gym.envs.registration import EnvRegistry, EnvSpec, registry
from sequoia.common.gym_wrappers.pixel_observation import PixelObservationWrapper

from .variant_spec import EnvVariantSpec


def add_pixel_variants(env_registry: EnvRegistry = registry) -> None:
    """ Adds pixel variants for the classic-control envs to the given registry in-place.
    """
    classic_control_env_specs: Dict[str, EnvSpec] = {
        spec.id: spec
        for env_id, spec in registry.env_specs.items()
        if isinstance(spec.entry_point, str)
        and spec.entry_point.startswith("gym.envs.classic_control")
    }

    for env_id, env_spec in classic_control_env_specs.items():
        new_id = "Pixel" + env_id
        if new_id not in env_registry.env_specs:
            new_spec = EnvVariantSpec.of(
                env_spec, new_id=new_id, wrappers=[PixelObservationWrapper]
            )
            env_registry.env_specs[new_id] = new_spec
