import warnings
from functools import singledispatch
from typing import Any, Dict, List, Type, Union, Callable

import inspect
import gym
from sequoia.common.gym_wrappers.multi_task_environment import make_env_attributes_task
from sequoia.settings.rl.envs import MUJOCO_INSTALLED
from gym.envs.registration import load, EnvRegistry, EnvSpec, registry, spec
import numpy as np
from gym.envs.classic_control import (
    CartPoleEnv,
    PendulumEnv,
    MountainCarEnv,
    Continuous_MountainCarEnv,
    AcrobotEnv,
)

# TODO: Add envs from PyBullet and from Box2d!
# from gym.envs.box2d import BipedalWalker, BipedalWalkerHardcore


class EnvironmentNotSupportedError(gym.error.UnregisteredEnv):
    """ Error raised when we don't know how to create a task for the given environment.
    """


# Idea: Create a true 'Task' class?
Task = Union[Dict[str, Any], Any]


@singledispatch
def make_task_for_env(
    env: gym.Env, step: int, change_steps: List[int], seed: int = None, **kwargs,
) -> Task:
    """ Generic function used by Sequoia's RL settings to create a "task" that will be
    applied to an environment like `env`.
    
    To add support for a new type of environment, simply register a handler function:

    ```
    @make_task_for_env.register(SomeGymEnvClass)
    def make_task_for_my_env(env: SomeGymEnvClass, step: int, change_steps: List[int], **kwargs,):
        return {"my_attribute": random.random()}
    ```

    NOTE: In order to create tasks for an environment through its string 'id', and to
    avoid having to actually instantiate an environment, `env` could perhaps be a type
    of environment rather than an actual environment instance. If your function can't
    handle this (raises an exception somehow), then a temporary environment will be
    created, and a warning will be raised.
    """
    raise NotImplementedError(f"Don't currently know how to create tasks for env {env}")


def is_supported(env_id: str, env_registry: EnvRegistry = registry) -> bool:
    env_spec = env_registry.spec(env_id)
    if callable(env_spec.entry_point):
        # assert False, (env_id, spec.entry_point, make_task_for_env.registry)
        # TODO: This won't work for IncrementalRLSetting if we call the `make_continuous_task_for_env` within the
        # `make_task_for_env` of the incremental/tasks.py
        return make_task_for_env.dispatch(
            env_spec.entry_point
        ) is not make_task_for_env.dispatch(object)
    assert isinstance(env_spec.entry_point, str)
    class_name = env_spec.entry_point.rsplit(":")[-1]
    if class_name in [c.__name__ for c in make_task_for_env.registry]:
        return True
    entry_point = load(env_spec.entry_point)
    if inspect.isfunction(entry_point):
        return False
    return make_task_for_env.dispatch(
        load(env_spec.entry_point)
    ) is not make_task_for_env.dispatch(object)


@make_task_for_env.register(type)
def make_task_for_env_type(env_type: Type[gym.Env], **kwargs) -> Task:
    try:
        # Try to create a task without actually instantiating the env, by passing the
        # type of env as the 'env' argument, rather than an env instance.
        env_handler_function = make_task_for_env.dispatch(env_type)
        return env_handler_function(env_type, **kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to create a task based only on the env type {env_type}: {exc}\n"
        ) from exc


@make_task_for_env.register(str)
def make_task_for_env_by_id(env: str, **kwargs,) -> Union[Dict[str, Any], Any]:
    # Load the entry-point class, and use it to determine what handler to use.
    # TODO: Actually instantiate the env here? or just dispatch based on the env class?
    if env not in registry.env_specs:
        raise RuntimeError(
            f"Can't create a task for env id {env}, since it isn't a registered env id."
        )
    env_spec: EnvSpec = registry.env_specs[env]
    env_entry_point: Callable[..., gym.Env] = load(env_spec.entry_point)
    # import inspect

    try:
        task: Task = make_task_for_env_type(env_entry_point, **kwargs)
        return task

    except RuntimeError as exc:
        warnings.warn(
            RuntimeWarning(
                f"A temporary environment will have to be created in order to make a task: {exc}"
            )
        )

    with gym.make(env) as temp_env:
        # IDEA: Could avoid re-creating the env between calls to this function, for
        # instance by saving a single temp env in a global variable and overwriting
        # it if `env` is of a different type.
        return make_task_for_env(temp_env, **kwargs)


@make_task_for_env.register
def make_task_for_wrapped_env(
    env: gym.Wrapper, step: int, change_steps: List[int] = None, **kwargs,
) -> Union[Dict[str, Any], Any]:
    # NOTE: Not sure if this is totally a good idea...
    # If someone registers a handler for some kind of Wrapper, than all envs wrapped
    # with that wrapper will use that handler, instead of their base environment type.
    return make_task_for_env(env.env, step=step, change_steps=change_steps, **kwargs)


# Dictionary mapping from environment type to a dict of environment values which can be
# modified with multiplicative gaussian noise.
_ENV_TASK_ATTRIBUTES: Dict[Union[Type[gym.Env]], Dict[str, float]] = {
    CartPoleEnv: {
        "gravity": 9.8,
        "masscart": 1.0,
        "masspole": 0.1,
        "length": 0.5,
        "force_mag": 10.0,
        "tau": 0.02,
    },
    PendulumEnv: {
        "max_speed": 8,
        "max_torque": 2.0,
        # "dt" = .05
        "g": 10.0,
        "m": 1.0,
        "l": 1.0,
    },
    MountainCarEnv: {
        "gravity": 0.0025,
        "goal_position": 0.45,  # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        "goal_velocity": 0,
    },
    Continuous_MountainCarEnv: {
        "goal_position": 0.45,  # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        "goal_velocity": 0,
    },
    # TODO: Test AcrobotEnv
    AcrobotEnv: {
        "LINK_LENGTH_1": 1.0,  # [m]
        "LINK_LENGTH_2": 1.0,  # [m]
        "LINK_MASS_1": 1.0,  #: [kg] mass of link 1
        "LINK_MASS_2": 1.0,  #: [kg] mass of link 2
        "LINK_COM_POS_1": 0.5,  #: [m] position of the center of mass of link 1
        "LINK_COM_POS_2": 0.5,  #: [m] position of the center of mass of link 2
        "LINK_MOI": 1.0,  #: moments of inertia for both links
    },
    # TODO: Add more of the classic control envs here.
    # TODO: Need to get the attributes to modify in each environment type and
    # add them here.
    # AtariEnv: [
    #     # TODO: Maybe have something like the difficulty as the CL 'task' ?
    #     # difficulties = temp_env.ale.getAvailableDifficulties()
    #     # "game_difficulty",
    # ],
}


@make_task_for_env.register(CartPoleEnv)
@make_task_for_env.register(PendulumEnv)
@make_task_for_env.register(MountainCarEnv)
@make_task_for_env.register(Continuous_MountainCarEnv)
def make_task_for_classic_control_env(
    env: gym.Env,
    step: int,
    change_steps: List[int] = None,
    task_params: Union[List[str], Dict[str, Any]] = None,
    seed: int = None,
    rng: np.random.Generator = None,
    noise_std: float = 0.2,
):
    # NOTE: `step` doesn't matter here, all tasks are independant.
    task_params = task_params or _ENV_TASK_ATTRIBUTES[type(env.unwrapped)]
    if step == 0:
        # Use the 'default' task as the first task.
        return task_params.copy()

    # Default back to the 'env attributes' task, which multiplies the default values
    # with normally distributed scaling coefficient.
    return make_env_attributes_task(
        env, task_params=task_params, seed=seed, rng=rng, noise_std=noise_std,
    )


# IDEA: Could probably not have these big ugly IF statements since we have the stubs for
# the different mujoco env classes anyway.

if MUJOCO_INSTALLED:
    from sequoia.settings.rl.envs.mujoco import ModifiedGravityEnv, HalfCheetahEnv

    default_mujoco_gravity = -9.81

    @make_task_for_env.register
    def make_task_for_modified_gravity_env(
        env: ModifiedGravityEnv,
        step: int,
        change_steps: List[int],
        seed: int = None,
        **kwargs,
    ) -> Union[Dict[str, Any], Any]:
        step_seed = seed * step if seed is not None else None
        # NOTE: np.random.default_rng(None) will NOT give the same result every first
        # time it is called, so this won't cause any issues with the same gravity being
        # sampled for all tasks if `seed` is None.
        rng = np.random.default_rng(step_seed)
        if step == 0:
            coefficient = 1
        else:
            coefficient = rng.uniform() + 0.5
        # TODO: Do we want to start with normal gravity?
        gravity = coefficient * default_mujoco_gravity
        return {"gravity": gravity}
