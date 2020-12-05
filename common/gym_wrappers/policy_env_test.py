from typing import List

import gym
import pytest
from gym.spaces import Discrete

from conftest import DummyEnvironment
from .policy_env import PolicyEnv, StateTransition


def test_iterating_with_policy():
    env = DummyEnvironment()
    env = PolicyEnv(env)
    env.seed(123)

    actions = [0, 1, 1, 2, 1, 1, 1, 1]
    expected_obs = [0, 0, 1, 2, 1, 2, 3, 4, 5]
    expected_rewards = [5, 4, 3, 4, 3, 2, 1, 0]
    expected_dones = [False, False, False, False, False, False, False, True]
    
    # Expect the transitions to have this form.
    expected_transitions = list(zip(expected_obs[0:],
                                    actions[0:],
                                    expected_obs[1:]))

    reset_obs = 0
    # obs = env.reset()
    # assert obs == reset_obs

    n_calls = 0
    def custom_policy(observations, action_space):
        # Deteministic policy used for testing purposes.
        nonlocal n_calls
        action = actions[n_calls]
        n_calls += 1
        return action

    n_expected_transitions = len(actions)
    env.set_policy(custom_policy)
    actual_transitions: List[StateTransition] = []

    i = 0
    for i, batch in enumerate(env):
        print(f"Step {i}: batch: {batch}")
        state_transition, reward = batch
        actual_transitions.append(state_transition)

        observation, action, next_observation = state_transition[:]

        assert observation == expected_obs[i]
        assert next_observation == expected_obs[i+1]
        assert action == actions[i]
        assert reward == expected_rewards[i]

    assert i == n_expected_transitions - 1
    assert len(actual_transitions) == n_expected_transitions
    assert [v.as_tuple() for v in actual_transitions] == expected_transitions
