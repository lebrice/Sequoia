
from sequoia.common.metrics.rl_metrics import EpisodeMetrics
from sequoia.conftest import DummyEnvironment

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
    env = EnvDataset(env)

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


def test_measure_RL_performance_batched_env():
    batch_size = 3
    start =[0 for i in range(batch_size)]
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
        action = np.ones(batch_size)
        reward = env.send(action)
        # print(obs, reward, done, info)
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
