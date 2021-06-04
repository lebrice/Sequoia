""" TODO: Add the tasks for IncrementalRLSetting, on top of the existing tasks from
ContinualRL
"""
from functools import singledispatch
import operator
import warnings
from typing import Any, Dict, List, Optional, Union, Callable
import gym
import numpy as np
from sequoia.settings.rl.continual.tasks import (
    make_task_for_env as make_continuous_task_for_env,
)
from sequoia.settings.rl.envs import (
    METAWORLD_INSTALLED,
    MONSTERKONG_INSTALLED,
    MTENV_INSTALLED,
    MUJOCO_INSTALLED,
    MetaMonsterKongEnv,
    MetaWorldEnv,
    MetaWorldMujocoEnv,
    SawyerXYZEnv,
    MTEnv,
)


DiscreteTask = Union[Dict[str, Any], Callable[[gym.Env], Any]]


@singledispatch
def make_task_for_env(
    env: gym.Env, step: int, change_steps: List[int], seed: int = None, **kwargs,
) -> DiscreteTask:
    """ Generic function used by Sequoia's IncrementalRL setting (and its descendants)
    to create a "task" that will be applied to an environment like `env`.

    To add support for a new type of environment, simply register a handler function:

    ```
    @make_task_for_env.register(SomeGymEnvClass)
    def make_task_for_my_env(env: SomeGymEnvClass, step: int, **kwargs,):
        return {"my_attribute": random.random()}
    ```
    """
    return make_continuous_task_for_env(
        env, step=step, change_steps=change_steps, seed=seed, **kwargs
    )
    # raise NotImplementedError(f"Don't currently know how to create a discrete task for env {env}")


# # Update the registry for this `make_task_for_env` function so it also includes the 
# for key, value in make_continuous_task_for_env.registry.items():
#     make_task_for_env.register(key, value)


if MONSTERKONG_INSTALLED:

    @make_task_for_env.register
    def make_task_for_monsterkong_env(
        env: MetaMonsterKongEnv,
        step: int,
        change_steps: List[int] = None,
        seed: int = None,
        **kwargs,
    ) -> Union[Dict[str, Any], Any]:
        """ Samples a task for the MonsterKong environment.

        TODO: When given a seed, sample the task randomly (but deterministicly) using
        the seed.
        """
        assert (
            change_steps is not None
        ), "Need task boundaries to construct the task schedule."

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


if MTENV_INSTALLED:

    @make_task_for_env.register
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

    @make_task_for_env.register(SawyerXYZEnv)
    @make_task_for_env.register(MetaWorldMujocoEnv)
    @make_task_for_env.register(MetaWorldEnv)
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
