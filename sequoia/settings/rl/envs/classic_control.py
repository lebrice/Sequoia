""" Registers variants of the classic-control envs that are used by sequoia. """
# TODO: Add Pixel???-v? variants for the classic-control envs.
from typing import Any, Callable, Dict, List, Optional, Union

import gym
from gym.envs.registration import EnvRegistry, EnvSpec, load, register, registry
from sequoia.common.gym_wrappers.pixel_observation import PixelObservationWrapper


def env_spec_variant(
    original_env_spec: EnvSpec,
    *,
    new_id: str,
    overwrite_kwargs: Dict[str, Any] = None,
    additional_wrappers: Optional[List[Callable[[gym.Env], gym.Env]]] = None,
) -> EnvSpec:
    """ Returns a new env spec which uses additional wrappers. """

    new_spec_kwargs = original_env_spec._kwargs
    new_spec_kwargs.update(overwrite_kwargs or {})

    new_entry_point: Union[str, Callable[..., gym.Env]] = original_env_spec.entry_point
    if additional_wrappers:
        # Get the callable that creates the env.
        if callable(original_env_spec.entry_point):
            env_fn = original_env_spec.entry_point
        else:
            env_fn = load(original_env_spec.entry_point)

        def _new_entry_point(**kwargs) -> gym.Env:
            env = env_fn(**kwargs)
            for wrapper in additional_wrappers:
                env = wrapper(env)
            return env

        new_entry_point = _new_entry_point

    return EnvSpec(
        new_id,
        entry_point=new_entry_point,
        reward_threshold=original_env_spec.reward_threshold,
        nondeterministic=original_env_spec.nondeterministic,
        max_episode_steps=original_env_spec.max_episode_steps,
        kwargs=new_spec_kwargs,
    )


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
            new_spec = env_spec_variant(
                env_spec, new_id=new_id, additional_wrappers=[PixelObservationWrapper]
            )
            env_registry.env_specs[new_id] = new_spec
