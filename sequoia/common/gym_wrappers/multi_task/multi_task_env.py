""" TODO: A special kind of gym.Wrapper that accepts a list of environments or
environment creating functions, and can swap between them when desired.
"""
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import gym
from gym import Wrapper, spaces
from functools import partial
from ..utils import IterableWrapper
from .add_task_labels import add_task_labels
from .tasks import Task, TaskType, ChangeEnvAttributes, get_changeable_attributes

EnvOrEnvFn = Union[gym.Env, Callable[..., gym.Env]]


class MultiTaskEnv(IterableWrapper):
    def __init__(self, env: Union[gym.Env, List[EnvOrEnvFn]], tasks: List[Task] = None):
        """Creates a Wrapper that can alternate between environments of different tasks.

        This can be either be passed a list of environments to use for each task, or a
        a single environment along with a list of tasks to apply to it when
        transitioning to the corresponding task.

        NOTE: The `env` argument can also be a list of functions that each return the
        environemnt to use for each task, rather than envs objects themselves. This is
        recommended especially when task switches will not occur often.

        NOTE: When passing a single env and a list of tasks, the env instance will be
        shared accross all tasks.

        Parameters
        ----------
        env : Union[gym.Env, List[EnvOrEnvFn]]
            Either a single env, or a list of List of `gym.Env` objects or of callables
            which produce an environment when called with no arguments.
            If `env` is just a single `gym.Env`, this wrapper has pretty much no effect.

        tasks : List[Task], optional
            When `env` is a single gym.Env, this list of tasks is used to create a list
            of 'fake' envs for each task, which will just consist in applying the envs.
            by default None

        ## Examples:

        - Using a list of environments:

        >>> import gym
        >>> from gym.envs.classic_control import CartPoleEnv
        >>> from functools import partial
        >>> def env_fn(i: int) -> CartPoleEnv:
        ...     env = gym.make("CartPole-v0")
        ...     # Change the length of the pole.
        ...     env.unwrapped.length = 0.5 + 0.1 * i
        ...     return env
        ...
        >>> multi_task_cartpole = MultiTaskEnv([partial(env_fn, i) for i in range(10)])
        >>> multi_task_cartpole.length
        0.5
        >>> multi_task_cartpole.change_task(1)
        >>> multi_task_cartpole.length
        0.6
        
        - Using a single env and a list of tasks:

        >>> import gym
        >>> from gym.envs.classic_control import CartPoleEnv
        >>> from .tasks import ChangeEnvAttributes
        >>> tasks = [ChangeEnvAttributes(length=0.5 + 0.1 * i) for i in range(10)]
        >>> multi_task_cartpole = MultiTaskEnv(gym.make("CartPole-v0"), tasks)
        >>> multi_task_cartpole.length
        0.5
        >>> multi_task_cartpole.change_task(1)
        >>> multi_task_cartpole.length
        0.6
        """
        self._envs: List[Union[gym.Env], Callable[..., gym.Env]] = []
        # TODO: Should we always require env constructors rather than envs themselves?
        if isinstance(env, gym.Env):
            tasks = tasks or [ChangeEnvAttributes(get_changeable_attributes(env))]
            env = [partial(task, env) for task in tasks]

        self.tasks: List[Task] = tasks or []
        for task_env in env:
            self._envs.append(task_env)

        env = self.get_env(0)
        super().__init__(env=env)

        self._current_task_index: int = 0
        self._seeds: List[Optional[int]] = []
        self._env_task_ids: List[int] = list(range(self.nb_tasks))


        task_label_space = spaces.Discrete(self.nb_tasks)
        self.observation_space = add_task_labels(
            self.env.observation_space, task_label_space
        )

    @property
    def nb_tasks(self) -> int:
        return len(self._envs)

    def get_env(self, task_index: int) -> gym.Env:
        """ Gets the environment at the given task index, creating it if necessary.

        If the envs passed to the constructor were constructors rather than gym.Env
        objects, the constructor for the given env will be called.
        """
        if isinstance(self._envs[task_index], gym.Env):
            return self._envs[task_index]
        return self._envs[task_index]()
    
    @property
    def current_task(self):
        return self._current_task_index
    
    def change_task(self, new_task_index: int) -> None:
        assert 0 <= new_task_index < self.nb_tasks

        # TODO: Do we want to close envs on switching tasks? or not?
        # self.env.close()
        self._current_task_index = new_task_index

        self.env = self.get_env(new_task_index)
        # TODO: Assuming the observations/action spaces don't change between tasks.

        if self._seeds and not self._using_live_envs:
            # Seed when creating the env, since we couldn't seed the env instance.
            self.env.seed(self._seeds[self._current_task_index])

        # self.action_space = self.env.action_space
        # self.observation_space = self.env.observation_space
        # self.reward_range = self.env.reward_range
        # self.metadata = self.env.metadata

        # How do we want to manage the 'reset' needed when switching tasks?
        # self._done = True
        # self._info["task_switch"] = True

    def observation(self, observation):
        """ Adds the id of the current task to the observations. """
        current_task_id = self._current_task_index
        if self._env_task_ids:
            current_task_id = self._env_task_ids[self._current_task_index]
        return add_task_labels(observation, current_task_id)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        return obs, reward, done, info

    def reset(self):
        return super().reset()

    def close(self):
        # TODO: Do we close just the current env, or all the envs?
        return super().close()

    def seed(self, seed: Optional[int]):
        # TODO: Do we seed just the current env, or all the envs?
        return self.env.seed()

    def seed_all(
        self, seeds: Union[Optional[int], List[Optional[int]]]
    ) -> List[Optional[int]]:
        """ Seeds all the envs (when possible) and return the seeds used in each. """
        if seeds is None or isinstance(seeds, int):
            seeds = [seeds for _ in range(self.nb_tasks)]

        assert len(seeds) == len(self._envs), "need as many seeds as envs"
        self._seeds = seeds

        if self._using_live_envs:
            results = []
            for env, seed in zip(self._envs, seeds):
                results.append(env.seed(seed))
                env.action_space.seed(seed)
                env.observation_space.seed(seed)
            return results
        else:
            # We can seed the 'live'/current env, but the others may be 'dormant', so we
            # just store the seeds, and we'll use them when 'waking up' the env.
            self._seeds = seeds
            self.seed(seeds[self._current_task_index])
            return self._seeds

    @property
    def _using_live_envs(self) -> bool:
        """Returns True if all the envs passed to this wrapper are 'live' gym.Env
        instances.

        Used internally just to know if we can directly manipulate the envs, or if we
        need to re-create them when using them.

        Returns
        -------
        bool
            True if all envs are 'live' `gym.Env` instances.
        """
        return all(isinstance(env, gym.Env) for env in self._envs)

    def __iter__(self):
        yield self.reset()
        for batch in super().__iter__():
            yield self.batch(batch)
