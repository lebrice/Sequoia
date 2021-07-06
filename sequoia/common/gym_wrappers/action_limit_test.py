import gym
from .action_limit import ActionLimit
from typing import List
import pytest
from sequoia.common.gym_wrappers.env_dataset import EnvDataset
from gym.wrappers import TimeLimit


def test_basics():
    env = gym.make("CartPole-v0")
    env = ActionLimit(env, max_steps=10)


def test_EnvDataset_of_ActionLimit():
    max_episode_steps = 10
    max_steps = 100
    env = gym.make("CartPole-v0")
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    env = ActionLimit(env, max_steps=max_steps)
    env = EnvDataset(env)
    done = False
    episode_steps: List[int] = []
    for episode in range(10):
        print(f"Staring episode {episode}, env.is_closed(): {env.is_closed()}")
        step = 0
        for step, obs in enumerate(env):
            print(f"Episode {episode}, Step {step}, obs {obs} {env.is_closed()}")
            assert step <= max_episode_steps
            env.send(env.action_space.sample())
        assert step > 0
        # NOTE: Here we have the last 'step' as 9.
        episode_steps.append(step)

    assert env.is_closed()
    assert sum(step + 1 for step in episode_steps) == max_steps


def test_ActionLimit_of_EnvDataset():
    max_episode_steps = 10
    max_steps = 100
    env = gym.make("CartPole-v0")
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    env = EnvDataset(env)
    env = ActionLimit(env, max_steps=max_steps)
    done = False
    episode_steps: List[int] = []
    for episode in range(10):
        print(f"Staring episode {episode}, env.is_closed(): {env.is_closed()}")
        step = 0
        for step, obs in enumerate(env):
            print(f"Episode {episode}, Step {step}, obs {obs} {env.is_closed()}")
            assert step <= max_episode_steps
            env.send(env.action_space.sample())
        assert step > 0
        # NOTE: Here we have the last 'step' as 9.
        episode_steps.append(step)

    assert env.is_closed()
    assert sum(step + 1 for step in episode_steps) == max_steps


from sequoia.settings.sl.wrappers.measure_performance_test import with_is_last


@pytest.mark.xfail(
    reason=(
        "BUG: Why is the BaseMethod working fine on a `TraditionalRLSetting, but "
        "not on an IncrementalRLSetting? Seems like the 'max_steps' isn't enforced the "
        " same way in both somehow."
    )
)
def test_delayed_EnvDataset_of_ActionLimit():
    """ Same test as above, however introduce a delay (like what's happening in the pl.Trainer)
    between the items sent by the trainer and the rewards returned by the env.

    """

    max_episode_steps = 10
    max_steps = 100
    env = gym.make("CartPole-v0")
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    env = EnvDataset(env)
    env = ActionLimit(env, max_steps=max_steps)
    done = False

    episode_steps: List[int] = []
    for episode in range(10):
        print(f"Staring episode {episode}, env.is_closed(): {env.is_closed()}")
        step = 0
        for step, (obs, is_last) in enumerate(with_is_last(env)):
            print(f"Episode {episode}, Step {step}, obs {obs} {env.is_closed()}")
            assert step <= max_episode_steps
            env.send(env.action_space.sample())
            if step == max_episode_steps - 1:
                assert is_last
        assert step > 0
        # NOTE: Here we have the last 'step' as 9.
        episode_steps.append(step)

    assert env.is_closed()
    assert sum(step + 1 for step in episode_steps) == max_steps

