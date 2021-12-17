from typing import Generic, Generator, Iterable

from abc import abstractmethod
import collections
import itertools
from typing import (
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Mapping,
    Optional,
    TypeVar,
    MutableSequence,
    Union,
    overload,
    runtime_checkable,
)
import gym
from dataclasses import dataclass, field, replace
from typing import Generator, Sequence, Tuple, Protocol, Any, Iterator, Type
import numpy as np
from enum import Enum, auto

# from typed_gym import Env, Space, VectorEnv
from functools import singledispatch
from sequoia.utils.generic_functions import detach, stack, get_slice

from .episode import (
    Episode,
    Observation,
    Observation_co,
    Action,
    Reward,
    Reward_co,
    T,
    T_co,
)
from .update_strategy import (
    Policy,
    PolicyUpdateStrategy,
    do_nothing_strategy,
    detach_actions_strategy,
    redo_forward_pass_strategy,
)

from gym.vector import VectorEnv
from sequoia.settings.base.environment import Environment


class EpisodeCollector(
    Generator[
        Episode[Observation, Action, Reward],
        Optional[Policy[Observation, Action]],
        None,
    ]
):
    def __init__(
        self,
        env: Union[
            Environment[Observation, Action, Reward],
            "VectorEnv[Observation, Action, Reward]",
        ],
        policy: Policy[Observation, Action],
        max_steps: Optional[int] = None,
        max_episodes: Optional[int] = None,
        what_to_do_after_update: PolicyUpdateStrategy = do_nothing_strategy,
    ) -> None:
        self.env = env
        self.policy = policy
        self.max_steps = max_steps
        self.max_episodes = max_episodes
        self.what_to_do_after_update = what_to_do_after_update

        self.total_steps = 0
        self.num_episodes: int = 0
        # List of *current*, *ongoing* episodes, has length `env.num_envs` for vectorenvs, or 1 for
        # envs.
        # NOT a buffer of episodes.
        self.ongoing_episodes: List[Episode[Observation, Action, Reward]] = []
        self._generator: Optional[
            Generator[
                Episode[Observation, Action, Reward],
                Optional[Policy[Observation, Action]],
                None,
            ]
        ] = None

        self.model_version: int = 0

    @property
    def generator(self):
        if self._generator is None:
            if isinstance(self.env, VectorEnv) or isinstance(
                self.env.unwrapped, VectorEnv
            ):
                self._generator = self._iter_vectorenv()
            else:
                self._generator = self._iter_env()
        return self._generator

    def __iter__(self):
        return self.generator

    def __next__(self):
        return next(self.generator)

    def send(self, new_policy: Optional[Policy[Observation, Action]]) -> None:  # type: ignore
        if new_policy is not None:
            self.on_policy_update(new_policy=new_policy)
        self.model_version += 1

    def throw(self, something):
        # TODO: No idea what to do here.
        return self.generator.throw(something)
        # raise something

    def _iter_env(self):
        """Yields full episodes from an environment."""
        for n_episodes in itertools.count():
            obs = self.env.reset()
            done = False
            # note, can't have a non-frozen dataclass if we want to set `last_observation` at
            # the end.
            episode: Episode[Observation, Action, Reward] = Episode(
                observations=[],
                actions=[],
                rewards=[],
                infos=[],
                last_observation=None,
                model_versions=[],
            )
            self.ongoing_episodes = [episode]

            while not done:
                # Get the action from the policy:
                action = self.policy(obs, action_space=self.env.action_space)

                episode.observations.append(obs)
                episode.actions.append(action)

                # Perform one step in the environment using that action.
                obs, reward, done, info = self.env.step(action)
                self.total_steps += 1

                episode.rewards.append(reward)
                episode.infos.append(info)
                episode.model_versions.append(self.model_version)

                if done:
                    # TODO: FrozenInstanceError if the Episode class is frozen!
                    object.__setattr__(episode, "last_observation", obs)
                    # episode.last_observation = obs

                    # Yield the episode, and if we get a new policy to use, then update it.
                    new_policy: Optional[Policy[Observation, Action]]
                    new_policy = yield episode.stack()
                    if new_policy:
                        # TODO: That doesn't really make sense in this case.. the episode will
                        # always be finished, since we only ever yield when an episode is
                        # finished..
                        if self.what_to_do_after_update is not do_nothing_strategy:
                            raise RuntimeError(
                                f"Can't really use the strategy {self.what_to_do_after_update} with a single env!"
                            )
                        # _ = what_to_do_after_update(unfinished_episodes=episode, old_policy=policy, new_policy=new_policy)
                        policy = new_policy

                    break  # Go to the next episode.

                if self.max_steps and self.total_steps >= self.max_steps:
                    print(f"Reached maximum number of steps. Breaking.")
                    break  # stop the current episode

            if self.max_episodes and n_episodes == self.max_episodes - 1:
                print(f"Reached max number of episodes.")
                break

    def _iter_vectorenv(
        self,
    ) -> Generator[
        Episode[Observation, Action, Reward],
        Optional[Policy[Observation, Action]],
        None,
    ]:
        """Generator that yields complete episodes from a vectorized environment.

        The episodes are flattened, so that they always just look like a regular episode from
        a single environment.
        """
        assert isinstance(self.env.unwrapped, VectorEnv), self.env
        num_envs = self.env.num_envs
        self.ongoing_episodes = [Episode() for _ in range(num_envs)]
        # IDEA: Create an iterator that uses the given policy and yields 'flattened' episodes.
        # NOTE: This is a lot easier to use with off-policy RL methods, for sure.
        observations: Observation = self.env.reset()
        dones: np.ndarray = np.zeros(num_envs, dtype=bool)

        for n_total_steps in itertools.count(step=num_envs):
            # Get an action for each environment from the policy.
            actions: Action = self.policy(
                observations, action_space=self.env.action_space
            )
            observations, rewards, dones, infos = self.env.step(actions)

            for i, (done, info) in enumerate(zip(dones, infos)):
                # NOTE: Using `get_slice` rather than just indexing, since they could be
                # something different, like a dict for example.
                observation: Any = get_slice(observations, i)
                # NOTE: THis 'actions' here, if we're using the 'detach_actions' update, it needs to also be detached, right?
                action = get_slice(actions, i)
                reward = get_slice(rewards, i)

                self.ongoing_episodes[i].actions.append(action)  # type: ignore
                self.ongoing_episodes[i].rewards.append(reward)  # type: ignore
                self.ongoing_episodes[i].infos.append(info)
                self.ongoing_episodes[i].model_versions.append(self.model_version)

            for i, (done, info) in enumerate(zip(dones, infos)):
                if done:
                    # A new policy to use can be passed to this generator via `send`.
                    # What to do with ongoing episodes is determined based on the value of
                    # `what_to_do_after_update`.
                    new_policy: Optional[Policy[Observation, Action]]
                    # NOTE: Not sure if we should stack or not.
                    new_policy = yield self.ongoing_episodes[i].stack()
                    self.num_episodes += 1

                    if new_policy is not None:
                        self.on_policy_update(new_policy)
                        # NOTE: Yielding nothing here, so send doesn't return anything.
                        yield  # type: ignore

                    # End of episode: reset the flag, put an empty new episode at that index.
                    dones[i] = False
                    self.ongoing_episodes[i] = Episode()

                    if self.max_episodes and self.num_episodes >= self.max_episodes:
                        print(f"Reached max episodes ({self.max_episodes}), stopping.")
                        # NOTE: Stopping here, rather than in the outside for loop, so that
                        # even if we have mroe than a single episode with `done=True`, we only
                        # yield the right number of episodes.
                        return

                # In both cases (both empty and non-empty Episode object), add the observation
                # to it.
                self.ongoing_episodes[i].observations.append(observation)

                # Assume that the individual envs don't close for now.
                if self.max_steps and n_total_steps >= self.max_steps:
                    print(f"Reached max total steps ({self.max_steps}), stopping.")
                    return

    def on_policy_update(self, new_policy: Policy[Observation, Action]):
        # print(
        #     f"Using a new policy starting from episode {num_episodes} ({total_steps=})"
        # )
        if isinstance(self.env, VectorEnv):
            len_before = [len(ep.actions) for ep in self.ongoing_episodes]
            self.ongoing_episodes = self.what_to_do_after_update(
                self.ongoing_episodes, old_policy=self.policy, new_policy=new_policy
            )
            if self.what_to_do_after_update in {
                detach_actions_strategy,
                redo_forward_pass_strategy,
            }:
                # print(f"Also detaching actions just in case.")
                actions = detach(actions)  # type: ignore

            len_after = [len(ep.actions) for ep in self.ongoing_episodes]
            assert len_before == len_after
            self.policy = new_policy
        else:
            # Doesn't really make sense, no?
            pass
