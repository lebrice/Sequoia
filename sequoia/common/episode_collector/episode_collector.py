import itertools

# from typed_gym import Env, Space, VectorEnv
from logging import getLogger as get_logger
from typing import Dict, Generator, List, Optional, Union

import numpy as np
from gym import Space
from gym.vector import VectorEnv
from sequoia.settings.base.environment import Environment
from sequoia.utils.generic_functions import get_slice
from sequoia.common.typed_gym import _Env, _VectorEnv
from .episode import Episode, StackedEpisode, _Action, _Observation, _Reward
from .update_strategy import (
    Policy,
    PolicyUpdateStrategy,
    detach_actions_strategy,
    do_nothing_strategy,
    redo_forward_pass_strategy,
)

logger = get_logger(__name__)


class EpisodeCollector(
    Generator[
        StackedEpisode[_Observation, _Action, _Reward],
        Optional[Policy[_Observation, _Action]],
        None,
    ]
):
    def __init__(
        self,
        env: Union[
            _Env[_Observation, _Action, _Reward], "_VectorEnv[_Observation, _Action, _Reward]",
        ],
        policy: Policy[_Observation, _Action],
        max_steps: Optional[int] = None,
        max_episodes: Optional[int] = None,
        what_to_do_after_update: PolicyUpdateStrategy[
            _Observation, _Action, _Reward
        ] = do_nothing_strategy,
        starting_model_version: int = 0,
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
        self.ongoing_episodes: List[Episode[_Observation, _Action, _Reward]] = []
        self._generator: Optional[
            Generator[
                StackedEpisode[_Observation, _Action, _Reward],
                Optional[Policy[_Observation, _Action]],
                None,
            ]
        ] = None

        self.model_version: int = starting_model_version

    @property
    def generator(self):
        if self._generator is None:
            if isinstance(self.env, VectorEnv) or isinstance(self.env.unwrapped, VectorEnv):
                self._generator = self._iter_vectorenv()
            else:
                self._generator = self._iter_env()
        return self._generator

    def __iter__(self):
        return self.generator

    def __next__(self):
        return next(self.generator)

    def send(self, new_policy: Optional[Policy[_Observation, _Action]]) -> None:  # type: ignore
        if new_policy is not None:
            self.generator.send(new_policy)
            # self.on_policy_update(new_policy=new_policy)
        # self.model_version += 1

    def throw(self, something):
        # TODO: No idea what to do here.
        return self.generator.throw(something)
        # raise something

    def close(self) -> None:
        self.generator.close()

    def _iter_env(self):
        """Yields full episodes from an environment."""
        for n_episodes in itertools.count():
            obs = self.env.reset()
            done = False
            # note, can't have a non-frozen dataclass if we want to set `last_observation` at
            # the end.
            episode: Episode[_Observation, _Action, _Reward] = Episode(
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
                # IDEA: How about we use the ID of the policy?
                # episode.model_versions.append(id(self.policy))

                if done:
                    # TODO: FrozenInstanceError if the Episode class is frozen!
                    object.__setattr__(episode, "last_observation", obs)
                    # episode.last_observation = obs

                    # Yield the episode, and if we get a new policy to use, then update it.
                    new_policy: Optional[Policy[_Observation, _Action]]
                    new_policy = yield episode.stack()
                    if new_policy:
                        self.model_version += 1
                        # NOTE: That doesn't really make sense in this case.. the episode will
                        # always be finished, since we only ever yield when an episode is
                        # finished..
                        # NOTE: Need to yield None so that `send` returns None.
                        yield

                        if self.what_to_do_after_update is not do_nothing_strategy:
                            logger.debug(
                                f"Can't really use the strategy {self.what_to_do_after_update} with a single env!"
                            )
                            # raise RuntimeError(
                            #     f"Can't really use the strategy {self.what_to_do_after_update} with a single env!"
                            # )
                        # _ = what_to_do_after_update(unfinished_episodes=episode, old_policy=policy, new_policy=new_policy)
                        self.policy = new_policy

                    break  # Go to the next episode.

                if self.max_steps and self.total_steps >= self.max_steps:
                    logger.debug(f"Reached maximum number of steps. Breaking.")
                    break  # stop the current episode

            if self.max_episodes and n_episodes == self.max_episodes - 1:
                logger.debug(f"Reached max number of episodes.")
                break

    def _iter_vectorenv(
        self,
    ) -> Generator[
        Episode[_Observation, _Action, _Reward], Optional[Policy[_Observation, _Action]], None,
    ]:
        """Generator that yields complete episodes from a vectorized environment.

        The episodes are flattened, so that they always just look like a regular episode from
        a single environment.
        """
        assert isinstance(self.env.unwrapped, VectorEnv), self.env
        num_envs: int = self.env.unwrapped.num_envs
        self.ongoing_episodes = [Episode() for _ in range(num_envs)]
        # IDEA: Create an iterator that uses the given policy and yields 'flattened' episodes.
        # NOTE: This is a lot easier to use with off-policy RL methods, for sure.
        observations: _Observation = self.env.reset()
        dones: np.ndarray = np.zeros(num_envs, dtype=bool)

        # The generator for the number of steps performed in this iteration. One is bounded, and the
        # other is not.
        if self.max_steps is None:
            step_range = itertools.count(step=num_envs)
        else:
            step_range = range(0, self.max_steps, num_envs)

        for n_total_steps in step_range:
            # Get an action for each environment from the policy.
            actions: _Action = self.policy(observations, action_space=self.env.action_space)
            observations, rewards, dones, infos = self.env.step(actions)

            # Quick loop for all the non-done episodes:
            # Dict mapping from environment index to the episode that have ended at this step.
            completed_episodes: Dict[int, StackedEpisode] = {}

            for env_idx, (done, info) in enumerate(zip(dones, infos)):
                observation = get_slice(observations, env_idx)
                action = get_slice(actions, env_idx)
                reward = get_slice(rewards, env_idx)

                # NOTE: Maybe we should yield a list of episodes that are done at each step?
                if not done:
                    # Episode isn't done yet. We can add the
                    self.ongoing_episodes[env_idx].observations.append(observation)
                    self.ongoing_episodes[env_idx].actions.append(action)  # type: ignore
                    self.ongoing_episodes[env_idx].rewards.append(reward)  # type: ignore
                    self.ongoing_episodes[env_idx].infos.append(info)
                    self.ongoing_episodes[env_idx].model_versions.append(self.model_version)
                else:
                    # Episode is done. Observation is the first of the next episode.
                    # BUG: `stack` actually has an issue with the terminal observation.
                    if "terminal_observation" in info:
                        last_observation = info["terminal_observation"]
                        self.ongoing_episodes[env_idx].last_observation = last_observation

                    # self.ongoing_episodes[env_idx].infos.append(info)
                    # TODO: Not sure about adding this or not:
                    # self.ongoing_episodes[env_idx].model_versions.append(self.model_version)
                    completed_episode = self.ongoing_episodes[env_idx].stack()
                    completed_episodes[env_idx] = completed_episode

                    # End of episode: Put an empty new episode at that index.
                    self.ongoing_episodes[env_idx] = Episode()
                    self.ongoing_episodes[env_idx].observations.append(observation)
                    self.ongoing_episodes[env_idx].actions.append(action)  # type: ignore
                    self.ongoing_episodes[env_idx].rewards.append(reward)  # type: ignore
                    self.ongoing_episodes[env_idx].infos.append({})
                    self.ongoing_episodes[env_idx].model_versions.append(self.model_version)

            if not completed_episodes:
                # No episode ended, go to the next step.
                continue

            single_action_space = getattr(self.env, "single_action_space", self.env.action_space)
            # Here's where we're at now:
            # We have *only* the ongoing episodes in `self.ongoing_episodes`, and *only* the
            # completed episodes in `completed_episodes`.
            # TODO: For each of these episodes, if we wanted to be clever about it, we'd actually
            # yield the list of episodes that are completed, and so we'd waste less on-policy data.
            # The 'batch size' of the DataLoader could then become the number of episodes per
            # update-ish?
            # TODO (better): Keep this as-is, and yield episodes one-at-a-time, but change the
            # OnPolicyModel so that instead of relying on `accumulate_grad_batches`, it instead has
            # the dataloader accumulate the episodes in a buffer before yielding them. Then, change
            # the training_step logic so that it calculates a loss using a list of epiodes.
            while completed_episodes:
                # NOTE: Using a while loop and popping items out, because we might modify
                # the following `completed_episodes` when a new policy comes up.
                # NOTE: pop order doesn't really matter, but going from lowest to highest.
                # env_idx, completed_episode = completed_episodes.popitem()
                logger.debug(f"Completed episodes to yield from envs: {completed_episodes.keys()}")
                logger.debug(f"{self.num_episodes=}: {self.max_episodes=}")

                env_idx = min(completed_episodes)
                completed_episode = completed_episodes.pop(env_idx)

                new_policy: Optional[Policy[_Observation, _Action]]

                if self.max_episodes and self.num_episodes >= self.max_episodes:
                    logger.info(f"Reached max episodes ({self.max_episodes}), stopping.")
                    # NOTE: Stopping here, rather than in the outside for loop, so that
                    # even if we have more than one episode with `done=True`, we only
                    # yield the right number of episodes in total.
                    return

                new_policy = yield completed_episode
                self.num_episodes += 1

                if new_policy is None:
                    # No policy update, yield the next completed episode if there is one.
                    continue

                # NOTE: Need to yield None here, so that `send` returns None.
                yield  # type: ignore

                # Increment this flag, which is left in the episode objects for convenience.
                self.model_version += 1

                assert self.what_to_do_after_update in {
                    redo_forward_pass_strategy,
                    do_nothing_strategy,
                    detach_actions_strategy,
                }
                # Update all the ongoing episodes:
                self.ongoing_episodes = self.what_to_do_after_update(
                    self.ongoing_episodes,
                    old_policy=self.policy,
                    new_policy=new_policy,
                    new_policy_version=self.model_version,
                    single_action_space=single_action_space,
                )
                # ALSO: Update all the completed episodes that haven't yet been yielded.
                updated_completed_episodes = self.what_to_do_after_update(
                    completed_episodes.values(),
                    old_policy=self.policy,
                    new_policy=new_policy,
                    new_policy_version=self.model_version,
                    single_action_space=single_action_space,
                )
                # Update the `completed_episodes` dict so that the next complete episodes to be
                # yielded are also updated correctly.
                completed_episodes = dict(
                    zip(completed_episodes.keys(), updated_completed_episodes)
                )
                # Update the `self.policy` attribute:
                self.policy = new_policy

    def on_policy_update(self, new_policy: Policy[_Observation, _Action]):
        # print(
        #     f"Using a new policy starting from episode {num_episodes} ({total_steps=})"
        # )
        if isinstance(self.env, VectorEnv):
            len_before = [len(ep.actions) for ep in self.ongoing_episodes]
            single_action_space: Space[_Action] = getattr(
                self.env, "single_action_space", self.env.action_space
            )
            self.ongoing_episodes = self.what_to_do_after_update(
                self.ongoing_episodes,
                old_policy=self.policy,
                new_policy=new_policy,
                single_action_space=single_action_space,
            )
            len_after = [len(ep.actions) for ep in self.ongoing_episodes]
            assert len_before == len_after
            self.policy = new_policy
        else:
            # Doesn't really make sense, no?
            pass
