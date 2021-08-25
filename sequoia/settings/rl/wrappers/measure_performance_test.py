
from sequoia.common.metrics.rl_metrics import EpisodeMetrics
from sequoia.conftest import DummyEnvironment
import pytest
from .measure_performance import MeasureRLPerformanceWrapper
from itertools import accumulate
# from sequoia.settings.rl.continual import ContinualRLSetting
from sequoia.common.gym_wrappers import EnvDataset
from gym.vector import SyncVectorEnv
import numpy as np
from functools import partial
import itertools


def test_measure_RL_performance_basics():
    env = DummyEnvironment(start=0, target=5, max_value=10)
    

    # env = TypedObjectsWrapper(env, observations_type=ContinualRLSetting.Observations, actions_type=ContinualRLSetting.Actions, rewards_type=ContinualRLSetting.Rewards)

    env = MeasureRLPerformanceWrapper(env)
    env.seed(123)
    all_episode_rewards = []
    all_episode_steps = []

    for episode in range(5):
        episode_steps = 0
        episode_reward = 0
        obs = env.reset()
        print(f"Episode {episode}, obs: {obs}")
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            episode_steps += 1
            # print(obs, reward, done, info)

        all_episode_steps.append(episode_steps)
        all_episode_rewards.append(episode_reward)
    from itertools import accumulate

    expected_metrics = {}
    for episode_steps, cumul_step, episode_reward in zip(all_episode_steps, accumulate(all_episode_steps), all_episode_rewards):
        expected_metrics[cumul_step] = EpisodeMetrics(
            n_samples=1,
            mean_episode_reward=episode_reward,
            mean_episode_length=episode_steps,
        )

    assert env.get_online_performance() == expected_metrics


def test_measure_RL_performance_iteration():
    env = DummyEnvironment(start=0, target=5, max_value=10)
    from gym.wrappers import TimeLimit
    max_episode_steps = 50
    env = EnvDataset(env)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)

    # env = TypedObjectsWrapper(env, observations_type=ContinualRLSetting.Observations, actions_type=ContinualRLSetting.Actions, rewards_type=ContinualRLSetting.Rewards)

    env = MeasureRLPerformanceWrapper(env)
    env.seed(123)
    all_episode_rewards = []
    all_episode_steps = []

    for episode in range(5):
        episode_steps = 0
        episode_reward = 0
        for step, obs in enumerate(env):
            print(f"Episode {episode}, obs: {obs}")
            action = env.action_space.sample()
            reward = env.send(action)
            episode_reward += reward
            episode_steps += 1
            # print(obs, reward, done, info)
            assert step <= max_episode_steps, "shouldn't be able to iterate longer than that."

        all_episode_steps.append(episode_steps)
        all_episode_rewards.append(episode_reward)

    expected_metrics = {}
    for episode_steps, cumul_step, episode_reward in zip(all_episode_steps, accumulate(all_episode_steps), all_episode_rewards):
        expected_metrics[cumul_step] = EpisodeMetrics(
            n_samples=1,
            mean_episode_reward=episode_reward,
            mean_episode_length=episode_steps,
        )

    assert env.get_online_performance() == expected_metrics


from typing import Union, Sequence
def done_is_true(done: Union[bool, np.ndarray, Sequence[bool]]) -> bool:
    return bool(done) if isinstance(done, bool) or not done.shape else all(done)

from sequoia.common.gym_wrappers.iterable_wrapper import IterableWrapper
class ProfiledActiveEnvironment(IterableWrapper):
    
    def iterator(self):
        obs = self.reset()
        steps = 0
        done = False
        while not done:
            action = yield steps, (obs, done_is_true(done))
            if action is None:
                action = self.action_
            assert action is not None, steps
            steps += 1
            obs, rewards, done, info = self.step(action)
            yield rewards

    def __iter__(self):
        self.__iterator = self.iterator()
        return self.__iterator

    def send(self, action):
        return self.__iterator.send(action)

        # for i, obs in enumerate(super().__iter__()):
        #     # logger.debug(f"Step {i}, obs.done={obs.done}")
        #     done = obs.done
        #     if not isinstance(done, bool) or not done.shape:
        #         # TODO: When we have batch size of 1, or more generally in RL, do we
        #         # want one call to `trainer.fit` to last a given number of episodes ?
        #         # TODO: Look into the `max_steps` argument to Trainer.
        #         done = all(done)
        #     # done = done or self.is_closed()
        #     done = self.is_closed()
        #     yield i, (obs, done)
def test_measure_RL_performance_iteration_with_profiler():
    env = DummyEnvironment(start=0, target=5, max_value=10)
    from gym.wrappers import TimeLimit
    max_episode_steps = 50
    env = EnvDataset(env)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)

    # env = TypedObjectsWrapper(env, observations_type=ContinualRLSetting.Observations, actions_type=ContinualRLSetting.Actions, rewards_type=ContinualRLSetting.Rewards)
    env = MeasureRLPerformanceWrapper(env)
    env = ProfiledActiveEnvironment(env)
    env.seed(123)
    all_episode_rewards = []
    all_episode_steps = []

    for episode in range(5):
        episode_steps = 0
        episode_reward = 0
        for step, (obs, is_last) in env:
            print(f"Episode {episode}, obs: {obs}, is_last: {is_last}")
            action = env.action_space.sample()
            reward = env.send(action)
            episode_reward += reward
            episode_steps += 1
            # print(obs, reward, done, info)
            assert step <= max_episode_steps, "shouldn't be able to iterate longer than that."

        all_episode_steps.append(episode_steps)
        all_episode_rewards.append(episode_reward)
    
    expected_metrics = {}
    for episode_steps, cumul_step, episode_reward in zip(all_episode_steps, accumulate(all_episode_steps), all_episode_rewards):
        expected_metrics[cumul_step] = EpisodeMetrics(
            n_samples=1,
            mean_episode_reward=episode_reward,
            mean_episode_length=episode_steps,
        )

    assert env.get_online_performance() == expected_metrics




@pytest.mark.xfail(
    reason=f"TODO: The wrapper seems to work but the test condition is too complicated"
)
def test_measure_RL_performance_batched_env():
    batch_size = 3
    start = [i for i in range(batch_size)]
    target = 5
    env = EnvDataset(SyncVectorEnv([
        partial(DummyEnvironment, start=start[i], target=target, max_value=target * 2)
        for i in range(batch_size)
    ]))
    # env = TypedObjectsWrapper(env, observations_type=ContinualRLSetting.Observations, actions_type=ContinualRLSetting.Actions, rewards_type=ContinualRLSetting.Rewards)

    env = MeasureRLPerformanceWrapper(env)
    env.seed(123)
    all_episode_rewards = []
    all_episode_steps = []

    for step, obs in enumerate(itertools.islice(env, 100)):
        print(f"step {step} obs: {obs}")
        action = np.ones(batch_size)  # always increment the counter
        reward = env.send(action)
        print(env.done_)
        # print(obs, reward, done, info)
    assert step == 99
    from collections import defaultdict
    from sequoia.common.metrics import Metrics

    expected_metrics = defaultdict(Metrics)
    for i in range(101):
        for env_index in range(batch_size):
            if i and i % target == 0:
                expected_metrics[i] += EpisodeMetrics(
                    n_samples=1,
                    mean_episode_reward=10., # ? FIXME: Actually understand this condition
                    mean_episode_length=target,
                )

            # FIXME: This test is a bit too complicated, hard to follow. I'll keep the
            # batches synced-up for now.
            # if i > 0 and (i + env_index) % target == 0:
            #     expected_metrics[i] += EpisodeMetrics(
            #         n_samples=1,
            #         mean_episode_reward=sum(target - (i + env_index % target) for j in range(start[env_index], target)),
            #         mean_episode_length=target - start[env_index] - 1
            #     )

    assert env.get_online_performance() == expected_metrics
