from abc import abstractmethod
from dataclasses import replace
from typing import List, Protocol

from gym import Space
from sequoia.utils.generic_functions import detach, stack

from .episode import Episode, _Action, _Observation_co, _Reward
from .policy import Policy


class PolicyUpdateStrategy(Protocol[_Observation_co, _Action, _Reward]):
    """Strategy for what to do with ongoing episodes when the policy is updated."""

    @abstractmethod
    def __call__(
        self,
        unfinished_episodes: List[Episode[_Observation_co, _Action, _Reward]],
        old_policy: Policy[_Observation_co, _Action],
        new_policy: Policy[_Observation_co, _Action],
        single_action_space: Space[_Action],
        new_policy_version: int,
    ) -> List[Episode[_Observation_co, _Action, _Reward]]:
        raise NotImplementedError


def do_nothing_strategy(
    unfinished_episodes: List[Episode], *args, **kwargs
) -> List[Episode]:
    return unfinished_episodes


def detach_actions_strategy(
    unfinished_episodes: List[Episode], *args, **kwargs
) -> List[Episode]:
    # Detaching stuff.
    return [
        replace(episode, actions=detach(episode.actions))
        for episode in unfinished_episodes
    ]


from sequoia.common.spaces.utils import batch_space


def redo_forward_pass_strategy(
    unfinished_episodes: List[Episode[_Observation_co, _Action, _Reward]],
    old_policy: Policy[_Observation_co, _Action],
    new_policy: Policy[_Observation_co, _Action],
    single_action_space: Space[_Action],
    new_policy_version: int,
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
        # TODO: Do we want to perform forward passes on everything?
        stacked_observations = stack(old_episode.observations)
        # Note: Need to create an action space that is consistent with the number of actions we need
        # from the policy.
        action_space = batch_space(single_action_space, n=len(old_episode.observations))
        new_actions = new_policy(stacked_observations, action_space=action_space)
        # NOTE: This simple 'len' check won't work if those are more complicated objects (e.g. dicts)
        # Might need to do something like a singledispatch unbatch function.
        assert len(old_episode.observations) == len(old_episode.actions), (len(old_episode.observations), len(old_episode.actions))
        assert len(new_actions) == len(old_episode.actions), (len(new_actions), len(old_episode.actions))
        # NOTE: What's happening is, the observations get added before the update hook is
        # called? so we're actually changing some stuff here!
        new_episode = replace(
            old_episode,
            actions=list(new_actions),
            model_versions=[new_policy_version for _ in old_episode.observations]
        )
        episodes.append(new_episode)
    return episodes
