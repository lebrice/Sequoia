""" TODO: Add the tasks for IncrementalRLSetting, on top of the existing tasks from
ContinualRL
"""
import operator
import warnings
from functools import singledispatch
from typing import Any, Callable, Dict, List, Optional, Union

import gym
from gym.envs.registration import registry, EnvRegistry, EnvSpec
import numpy as np
from sequoia.settings.rl.envs import (
    METAWORLD_INSTALLED,
    MONSTERKONG_INSTALLED,
    MTENV_INSTALLED,
    MUJOCO_INSTALLED,
    MetaMonsterKongEnv,
    MetaWorldEnv,
    MetaWorldMujocoEnv,
    MTEnv,
    SawyerXYZEnv,
)

from ..discrete.tasks import ContinuousTask, DiscreteTask
from ..discrete.tasks import is_supported as _is_supported
from ..discrete.tasks import make_discrete_task

IncrementalTask = DiscreteTask


@singledispatch
def make_incremental_task(
    env: gym.Env, step: int, change_steps: List[int], seed: int = None, **kwargs,
) -> IncrementalTask:
    """ Generic function used by Sequoia's `IncrementalRLSetting` (and its
    descendants) to create a "task" that will be applied to an environment like `env`.

    To add support for a new type of environment, simply register a handler function:
    ```
    @make_incremental_task.register(SomeGymEnvClass)
    def make_incremental_task_for_my_env(env: SomeGymEnvClass, step: int, change_steps: List[int], **kwargs,):
        return {"my_attribute": random.random()}
    ```
    """
    return make_discrete_task(
        env, step=step, change_steps=change_steps, seed=seed, **kwargs
    )

# IDEA: Could probably do something like this automatically using a function decorator.
# Update the registry for this `make_incremental_task` function so it builds on top of the
# existing 'make_discrete_task' in the ContinualRLSetting, since we can create a schedule
# of discrete tasks from a continuous one, but not the other way around.
for env_class, env_task_creation_func in make_incremental_task.registry.items():
    make_incremental_task.register(env_class, env_task_creation_func)


def is_supported(
    env_id: str,
    env_registry: EnvRegistry = registry,
    _make_discrete_task: Callable[..., DiscreteTask] = make_incremental_task,
) -> bool:
    """ Returns wether Sequoia is able to create (incremental) tasks for the given
    environment.
    """
    return _is_supported(
        env_id=env_id, env_registry=env_registry, _make_task_function=make_incremental_task,
    )


if MTENV_INSTALLED:

    @make_incremental_task.register
    def make_task_for_mtenv_env(
        env: MTEnv, step: int, change_steps: List[int], seed: int = None, **kwargs,
    ) -> Callable[[MTEnv], None]:
        """ Samples a task for an env from MTEnv.

        The Task in this case will be a callable that will call the env's
        `set_task_state` method, passing in an integer (`task`).

        When `seed` is None, then the task will be the same as the task index.
        """
        assert change_steps, "Need task boundaries to construct the task schedule."

        if step not in change_steps:
            raise RuntimeError(
                f"MTENV has discrete tasks (as far as I'm aware), so step {step} "
                f"should be in {change_steps}!"
            )

        task_index = change_steps.index(step)

        task_states = list(range(len(change_steps)))
        if seed is not None:
            # perform a deterministic shuffling of the 'task ids'
            rng = rng or np.random.default_rng(seed)
            rng.shuffle(task_states)

        # NOTE: Task state is an integer for now, but I'm not sure if it can also be
        # something else..
        task_state: int = task_states[task_index]
        return operator.methodcaller("set_task_state", task_state)


if METAWORLD_INSTALLED:

    @make_incremental_task.register(SawyerXYZEnv)
    @make_incremental_task.register(MetaWorldMujocoEnv)
    @make_incremental_task.register(MetaWorldEnv)
    def make_task_for_metaworld_env(
        env: MetaWorldEnv,
        step: int,
        change_steps: List[int] = None,
        seed: int = None,
        **kwargs,
    ) -> Callable[[MetaWorldEnv], None]:
        """Samples a task for an environment from MetaWorld.

        The Task in this case will be a callable that will call the env's
        `set_task` method, passing in a task from the `train_tasks` of the benchmark
        that contains this environment.

        When `seed` is None, then the task will be the same as the task index.
        """
        # TODO: Which benchmark should we use?
        found = False

        assert change_steps, "Need task boundaries to construct the task schedule."

        if step not in change_steps:
            raise RuntimeError(
                f"MTENV has discrete tasks (as far as I'm aware), so step {step} "
                f"should be in {change_steps}!"
            )

        task_index = change_steps.index(step)

        import metaworld

        # TODO: Not sure how exactly we're supposed to use the train_classes vs
        # train_tasks, should it be a MultiTaskEnv within a given env class?
        warnings.warn(RuntimeWarning("This is supposedly not the right way to do it!"))
        env_name = ""
        # Find the benchmark that contains this type of env.
        for benchmark_class in [metaworld.ML10]:
            benchmark = benchmark_class()
            for env_name, env_class in benchmark.train_classes.items():
                if isinstance(env, env_class):
                    # Found the right benchmark that contains this env class, now
                    # create the task schedule using
                    # the tasks.
                    found = True
                    break
            if found:
                break
        if not found:
            raise NotImplementedError(
                f"Can't find a benchmark with env class {type(env)}!"
            )
        # `benchmark` is here the right benchmark to use to create the tasks.
        training_tasks = [
            task for task in benchmark.train_tasks if task.env_name == env_name
        ]

        tasks = training_tasks.copy()
        if seed is not None:
            # perform a deterministic shuffling of the 'task ids'
            rng = rng or np.random.default_rng(seed)
            rng.shuffle(tasks)

        task = tasks[task_index]
        return operator.methodcaller("set_task", task)
