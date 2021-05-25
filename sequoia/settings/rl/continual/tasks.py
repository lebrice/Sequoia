import warnings
from functools import singledispatch
from typing import Any, Dict, List, Type, Union

import gym
from sequoia.common.gym_wrappers.multi_task_environment import \
    make_env_attributes_task
from sequoia.settings.rl.envs import MUJOCO_INSTALLED

import numpy as np
from gym.envs.classic_control import CartPoleEnv, PendulumEnv, MountainCarEnv, Continuous_MountainCarEnv
from gym.envs.box2d import BipedalWalker, BipedalWalkerHardcore 


@singledispatch
def make_task_for_env(
    env: gym.Env, step: int, change_steps: List[int] = None, **kwargs,
) -> Union[Dict[str, Any], Any]:
    # warnings.warn(
    #     RuntimeWarning(
    #         f"Don't yet know how to create a task for env {env}, will use the environment as-is."
    #     )
    # )
    # #FIXME: Remove this after debugging is done.
    # raise NotImplementedError(f"Don't currently know how to create tasks for env {env}")
    return {}


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
        "goal_position": 0.45, # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        "goal_velocity": 0,
    },
    Continuous_MountainCarEnv: {
        "goal_position": 0.45, # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        "goal_velocity": 0,
    }
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


if MUJOCO_INSTALLED:
    from sequoia.settings.rl.envs.mujoco import ModifiedGravityEnv

    _ENV_TASK_ATTRIBUTES[ModifiedGravityEnv] = {"gravity": -9.81}

    @make_task_for_env.register
    def make_task_for_modified_gravity_env(
        env: ModifiedGravityEnv,
        step: int,
        change_steps: List[int],
        seed: int = None,
        rng: np.random.Generator = None,
        **kwargs,
    ) -> Union[Dict[str, Any], Any]:
        rng = rng or np.random.default_rng(seed)
        coefficient = rng.uniform() + 0.5
        # TODO: Do we want to start with normal gravity?
        if step == 0:
            coefficient = 1
        gravity = coefficient * -9.81
        return {"gravity": gravity}
