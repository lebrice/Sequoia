from typing import Dict, Union, Type, List, Any
import gym

from sequoia.common.gym_wrappers.multi_task_environment import make_env_attributes_task
from functools import singledispatch
import warnings

@singledispatch
def make_task_for_env(
    env: gym.Env,
    step: int,
    change_steps: List[int] = None,
    **kwargs,
) -> Union[Dict[str, Any], Any]:
    # warnings.warn(
    #     RuntimeWarning(
    #         f"Don't yet know how to create a task for env {env}, will use the environment as-is."
    #     )
    # )
    # #FIXME: Remove this after debugging is done.
    # raise NotImplementedError(f"Don't currently know how to create tasks for env {env}")
    return {}


@make_task_for_env.register(gym.Wrapper)
def make_task_for_wrapped_env(
    env: gym.Wrapper,
    step: int,
    change_steps: List[int] = None,
    **kwargs,
) -> Union[Dict[str, Any], Any]:
    # warnings.warn(
    #     RuntimeWarning(
    #         f"Don't yet know how to create a task for env {env}, will use the environment as-is."
    #     )
    # )
    # #FIXME: Remove this after debugging is done.
    # raise NotImplementedError(f"Don't currently know how to create tasks for env {env}")
    return make_task_for_env(env.env, step=step, change_steps=change_steps, **kwargs)


import numpy as np
from gym.envs.classic_control import CartPoleEnv, PendulumEnv


_ENV_TASK_ATTRIBUTES: Dict[Union[Type[gym.Env]], Dict[str, float]] = {
    CartPoleEnv: {
        "gravity"  : 9.8,
        "masscart"  : 1.0,
        "masspole"  : 0.1,
        "length"  : 0.5,
        "force_mag": 10.0,
        "tau" : 0.02,
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

    return make_env_attributes_task(
        env,
        task_params=task_params,
        seed=seed,
        rng=rng,
        noise_std=noise_std,
    )
    