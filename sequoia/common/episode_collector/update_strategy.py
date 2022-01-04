from abc import abstractmethod
from dataclasses import replace
from functools import singledispatch
from logging import getLogger as get_logger
from typing import List, Protocol, TypeVar

import numpy as np
import torch
from gym import Space
from torch import Tensor

from sequoia.common.spaces.utils import batch_space
from sequoia.common.typed_gym import _Space
from sequoia.utils.generic_functions import detach, stack
from sequoia.utils.generic_functions.slicing import get_slice, set_slice
from sequoia.utils.generic_functions.stack import stack, unstack

from .episode import Episode, StackedEpisode, _Action, _Observation_co, _Reward
from .policy import Policy

logger = get_logger(__name__)


class PolicyUpdateStrategy(Protocol[_Observation_co, _Action, _Reward]):
    """Strategy for what to do with ongoing episodes when the policy is updated."""

    @abstractmethod
    def __call__(
        self,
        unfinished_episodes: List[Episode[_Observation_co, _Action, _Reward]],
        old_policy: Policy[_Observation_co, _Action],
        new_policy: Policy[_Observation_co, _Action],
        single_action_space: _Space[_Action],
        new_policy_version: int,
    ) -> List[Episode[_Observation_co, _Action, _Reward]]:
        raise NotImplementedError


def do_nothing_strategy(unfinished_episodes: List[Episode], *args, **kwargs) -> List[Episode]:
    return unfinished_episodes


def detach_actions_strategy(unfinished_episodes: List[Episode], *args, **kwargs) -> List[Episode]:
    # Detaching stuff.
    return [replace(episode, actions=detach(episode.actions)) for episode in unfinished_episodes]


_Episode = TypeVar("_Episode", Episode, StackedEpisode)


def redo_forward_pass_strategy(
    episodes: List[_Episode],
    new_policy: Policy[_Observation_co, _Action],
    single_action_space: _Space[_Action],
    new_policy_version: int,
) -> List[_Episode]:
    # NOTE: Using list(new_actions) so that we can use 'append' after.
    # Todo: Need to use a generic `stack` method for the inputs to the policy here.
    # TODO: This isn't trivial, does a forward pass on a batch, but instead of having the
    # batch dimension as one env each as usual, not its an episode in the same env!
    # return [
    #     replace(old_episode,
    #         actions=list(new_policy(stack(old_episode.observations))
    #     ))
    #     for old_episode in unfinished_episodes
    # ]

    # NOTE: Other (uglier) alternative:
    # old_episodes = episodes
    updated_episodes = []
    for old_episode in episodes:
        # NOTE: Could probably also only do the slice that needs to be recomputed!

        versions_differ = np.not_equal(old_episode.model_versions, new_policy_version)

        if all(versions_differ):
            new_episode = replace_actions(
                old_episode,
                new_policy=new_policy,
                single_action_space=single_action_space,
                new_policy_version=new_policy_version,
            )

        else:
            assert False, "todo: still debugging stuff here."
            old_indices = np.nonzero(versions_differ)[0]
            logger.debug(
                f"Only need to update indices {len(old_indices)} in the episode of length "
                f"{len(old_episode)}."
            )
            old_episode_slice = get_slice(old_episode, old_indices)

            new_episode_slice = replace_actions(
                old_episode_slice,
                old_policy=old_policy,
                new_policy=new_policy,
                single_action_space=single_action_space,
                new_policy_version=new_policy_version,
            )
            new_episode = replace(old_episode)
            model_versions_before = old_episode.model_versions.copy()
            set_slice(new_episode, old_indices, new_episode_slice)

            # TODO: Still debugging this branch here.
            assert False, (
                new_policy_version,
                versions_differ,
                old_indices,
                model_versions_before,
                new_episode.model_versions,
            )

        updated_episodes.append(new_episode)
    return updated_episodes


@singledispatch
def replace_actions(
    old_episode: Episode[_Observation_co, _Action, _Reward],
    new_policy: Policy[_Observation_co, _Action],
    single_action_space: _Space[_Action],
    new_policy_version: int,
) -> Episode[_Observation_co, _Action, _Reward]:
    stacked_observations = stack(old_episode.observations)
    # Need to create an action space that is consistent with the number of actions we need
    # from the policy.
    n_actions = len(old_episode.actions)
    batched_action_space = batch_space(single_action_space, n=n_actions)
    new_stacked_actions = new_policy(stacked_observations, action_space=batched_action_space)
    # NOTE: Use a generic 'unstack' function here.
    # Need to unstack the actions so that we can add more stuff to the episode after.
    # (Since in `Episode` objects have list fields.)
    new_actions = unstack(new_stacked_actions)
    # BUG: There's an issue here, due to the dtypes not being the same.
    new_episode = replace(
        old_episode,
        actions=new_actions,
        model_versions=[new_policy_version for _ in old_episode.observations],
    )
    assert len(new_actions) == len(old_episode.actions), (old_episode, new_episode)
    assert new_episode.observations is old_episode.observations
    assert new_episode.rewards is old_episode.rewards
    return new_episode


@replace_actions.register(StackedEpisode)
def _replace_stacked_actions(
    old_episode: StackedEpisode[_Observation_co, _Action, _Reward],
    new_policy: Policy[_Observation_co, _Action],
    single_action_space: _Space[_Action],
    new_policy_version: int,
) -> StackedEpisode[_Observation_co, _Action, _Reward]:

    stacked_observations = old_episode.observations
    # Need to create an action space that is consistent with the number of actions we need
    # from the policy.
    # TODO: How to reliably get the number of observations to use here, if obs isn't a list?
    n_actions = old_episode.length
    action_space = batch_space(single_action_space, n=n_actions)

    new_stacked_actions = new_policy(stacked_observations, action_space=action_space)
    # NOTE: Use a generic 'unstack' function here.
    # Need to unstack the actions so that we can add more stuff to the episode after.
    # (Since in `Episode` objects have list fields.)

    if isinstance(old_episode.model_versions, Tensor):
        new_model_versions = torch.ones_like(old_episode.model_versions) * new_policy_version
    else:
        new_model_versions = np.array([new_policy_version for _ in range(n_actions)])

    new_episode = replace(
        old_episode,
        actions=new_stacked_actions,
        model_versions=np.array([new_policy_version for _ in range(n_actions)]),
    )
    assert new_episode.observations is old_episode.observations
    assert new_episode.rewards is old_episode.rewards
    return new_episode
