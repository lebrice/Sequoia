
from abc import abstractmethod
from typing import (
    List,
)
from dataclasses import replace
from typing import Protocol
from sequoia.utils.generic_functions import detach, stack

from .episode import Episode, Observation_co, Action, Reward
from .policy import Policy



class PolicyUpdateStrategy(Protocol[Observation_co, Action, Reward]):
    """ Strategy for what to do with ongoing episodes when the policy is updated. """

    @abstractmethod
    def __call__(
        self,
        unfinished_episodes: List[Episode[Observation_co, Action, Reward]],
        old_policy: Policy[Observation_co, Action],
        new_policy: Policy[Observation_co, Action],
    ) -> List[Episode[Observation_co, Action, Reward]]:
        raise NotImplementedError


def do_nothing_strategy(
    unfinished_episodes: List[Episode], old_policy: Policy, new_policy: Policy
) -> List[Episode]:
    return unfinished_episodes


def detach_actions_strategy(
    unfinished_episodes: List[Episode], old_policy: Policy, new_policy: Policy
) -> List[Episode]:
    # Detaching stuff.
    return [
        replace(episode, actions=([detach(action) for action in episode.actions]))
        for episode in unfinished_episodes
    ]


def redo_forward_pass_strategy(
    unfinished_episodes: List[Episode], old_policy: Policy, new_policy: Policy
) -> List[Episode]:
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
    episodes = []
    for i, old_episode in enumerate(unfinished_episodes):
        if len(old_episode.actions) == len(old_episode.observations) - 1:
            stacked_obs = stack(old_episode.observations[:-1])
        else:
            stacked_obs = stack(old_episode.observations)

        new_actions = new_policy(stacked_obs)
        assert len(new_actions) == len(old_episode.actions)
        # NOTE: WHat's happening is, the observations get added before the update hook is
        # called? so we're actually changing some stuff here!
        new_episode = replace(old_episode, actions=list(new_actions))
        episodes.append(new_episode)
    return episodes

