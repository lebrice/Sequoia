""" Functions that create 'discrete' tasks for an environment. 

TODO: Once we have a wrapper that can seamlessly switch from one env to the next, then
move the "incremental" tasks from `incremental/tasks.py` to this level.
"""

import warnings
from functools import partial, singledispatch
from typing import Any, Callable, Dict, List, Optional, Union

import gym
import numpy as np

from sequoia.settings.rl.envs import MONSTERKONG_INSTALLED, MetaMonsterKongEnv, sequoia_registry

from ..continual.tasks import (
    ContinuousTask,
    _is_supported,
    make_continuous_task,
    task_sampling_function,
)

DiscreteTask = Union[ContinuousTask, Callable[[gym.Env], Any]]


@task_sampling_function(env_registry=sequoia_registry, based_on=make_continuous_task)
@singledispatch
def make_discrete_task(
    env: gym.Env,
    *,
    step: int,
    change_steps: List[int],
    seed: int = None,
    **kwargs,
) -> DiscreteTask:
    """Generic function used by Sequoia's `DiscreteTaskAgnosticRLSetting` (and its
    descendants) to create a "task" that will be applied to an environment like `env`.

    To add support for a new type of environment, simply register a handler function:

    ```
    @make_discrete_task.register(SomeGymEnvClass)
    def make_discrete_task_for_my_env(env: SomeGymEnvClass, step: int, change_steps: List[int], **kwargs,):
        return {"my_attribute": random.random()}
    ```
    """
    raise NotImplementedError(f"Don't currently know how to create a discrete task for env {env}")
    # return make_continuous_task(
    #     env, step=step, change_steps=change_steps, seed=seed, **kwargs
    # )


is_supported = partial(_is_supported, _make_task_function=make_discrete_task)


if MONSTERKONG_INSTALLED:
    # In MonsterKong the tasks can be changed on-the-fly, whereas they can't in the
    # size-based MUJOCO envs.

    @make_discrete_task.register
    def make_task_for_monsterkong_env(
        env: MetaMonsterKongEnv,
        step: int,
        change_steps: List[int] = None,
        seed: int = None,
        **kwargs,
    ) -> Union[Dict[str, Any], Any]:
        """Samples a task for the MonsterKong environment.

        TODO: When given a seed, sample the task randomly (but deterministicly) using
        the seed.
        """
        assert change_steps is not None, "Need task boundaries to construct the task schedule."

        if step not in change_steps:
            raise RuntimeError(
                f"Monsterkong's has discrete tasks, {step} should be in {change_steps}!"
            )
        task_index = change_steps.index(step)

        # TODO: double-check with @mattriemer on this:
        n_supported_levels = 30
        # IDEA: Could also have a list of supported levels
        levels = list(range(n_supported_levels))
        nb_tasks = len(change_steps)

        rng: Optional[np.random.Generator] = None
        if seed is not None:
            # perform a deterministic shuffling of the 'task ids'
            rng = np.random.default_rng(seed)
            rng.shuffle(levels)

        level: int
        if task_index >= n_supported_levels:
            warnings.warn(
                RuntimeWarning(
                    f"The given task id ({task_index}) is greater than the number of "
                    f"levels currently available in MonsterKong "
                    f"({n_supported_levels})!\n"
                    f"Multiple tasks may therefore use the same level!"
                )
            )
            # Option 1: Loop back around, using the same task as the first task?
            # (Probably not a good idea, since then we might get to train on the first
            # tasks right before testing begins! (which isnt great as a CL evaluation)
            # task_index %= n_supported_levels

            # Option 2 (better): Sample levels at random after all other levels have been
            # exhausted.
            # NOTE: Other calls to this should not get the same value!
            rng = rng or np.random.default_rng(seed)
            random_extra_levels = rng.integers(
                0, n_supported_levels, size=nb_tasks - n_supported_levels
            )
            level = int(random_extra_levels[task_index - n_supported_levels])
        else:
            level = levels[task_index]

        return {"level": level}
