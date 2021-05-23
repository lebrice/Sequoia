""" TODO: Tests for the 'measure performance wrapper' to be used to get the performance
over the first "epoch" 
"""
import dataclasses
import itertools
from functools import partial
from typing import Iterable, Tuple, TypeVar
from itertools import accumulate

import gym
import numpy as np
import pytest
import torch
from gym.vector import SyncVectorEnv
from torch.utils.data import TensorDataset

from sequoia.common import Config
from sequoia.common.gym_wrappers import AddDoneToObservation, EnvDataset
from sequoia.common.metrics import ClassificationMetrics
from sequoia.settings.rl.continual.wrappers import TypedObjectsWrapper
from sequoia.settings.rl.continual.setting import ContinualRLSetting
from sequoia.settings.sl.environment import PassiveEnvironment
from .setting import ClassIncrementalSetting
from .measure_performance_wrapper import MeasureSLPerformanceWrapper
from .objects import Actions, Observations, Rewards


def test_measure_performance_wrapper():
    dataset = TensorDataset(
        torch.arange(100).reshape([100, 1, 1, 1]) * torch.ones([100, 3, 32, 32]),
        torch.arange(100),
    )
    pretend_to_be_active = True
    env = PassiveEnvironment(
        dataset, batch_size=1, n_classes=100, pretend_to_be_active=pretend_to_be_active
    )
    for i, (x, y) in enumerate(env):
        # print(x)
        assert y is None if pretend_to_be_active else y is not None
        assert (x == i).all()
        action = i if i < 50 else 0
        reward = env.send(action)
        assert reward == i
    assert i == 99
    # This might be a bit weird, since .reset() will give the same obs as the first x
    # when iterating.
    obs = env.reset()
    for i, (x, y) in enumerate(env):
        # print(x)
        assert y is None
        assert (x == i).all()
        action = i if i < 50 else 0
        reward = env.send(action)
        assert reward == i
    assert i == 99

    env = TypedObjectsWrapper(
        env, observations_type=Observations, actions_type=Actions, rewards_type=Rewards
    )
    # TODO: Do we want to require Observations / Actions / Rewards objects?
    env = MeasureSLPerformanceWrapper(env, first_epoch_only=False)
    for epoch in range(3):
        for i, (observations, rewards) in enumerate(env):
            assert observations is not None
            assert rewards is None
            assert (observations.x == i).all()

            # Only guess correctly for the first 50 steps.
            action = Actions(y_pred=np.array([i if i < 50 else 0]))
            rewards = env.send(action)
            assert (rewards.y == i).all()
        assert i == 99
    assert epoch == 2

    assert set(env.get_online_performance().keys()) == set(range(100 * 3))
    for i, (step, metric) in enumerate(env.get_online_performance().items()):
        assert step == i
        assert metric.accuracy == (1.0 if (i % 100) < 50 else 0.0), (i, step, metric)

    metrics = env.get_average_online_performance()
    assert isinstance(metrics, ClassificationMetrics)
    # Since we guessed the correct class only during the first 50 steps.
    assert metrics.accuracy == 0.5


def make_dummy_env(n_samples: int = 100, batch_size: int = 1):
    dataset = TensorDataset(
        torch.arange(n_samples).reshape([n_samples, 1, 1, 1])
        * torch.ones([n_samples, 3, 32, 32]),
        torch.arange(n_samples),
    )
    pretend_to_be_active = False
    env = PassiveEnvironment(
        dataset,
        batch_size=batch_size,
        n_classes=n_samples,
        pretend_to_be_active=pretend_to_be_active,
    )
    env = TypedObjectsWrapper(
        env, observations_type=Observations, actions_type=Actions, rewards_type=Rewards
    )
    return env


def test_measure_performance_wrapper_first_epoch_only():
    env = make_dummy_env(n_samples=100, batch_size=1)
    env = MeasureSLPerformanceWrapper(env, first_epoch_only=True)

    for epoch in range(2):
        print(f"start epoch {epoch}")
        for i, (observations, rewards) in enumerate(env):
            assert observations is not None
            if epoch == 0:
                assert rewards is None
            else:
                assert rewards is not None
                rewards_ = rewards  # save these for a comparison below.

            assert (observations.x == i).all()

            # Only guess correctly for the first 50 steps.
            action = Actions(y_pred=np.array([i if i < 50 else 0]))

            rewards = env.send(action)
            if epoch != 0:
                # We should just receive what we already got by iterating.
                assert rewards.y == rewards_.y
            assert (rewards.y == i).all()
        assert i == 99

    # do another epoch, but this time don't even send actions.
    for i, (observations, rewards) in enumerate(env):
        assert (observations.x == i).all()
        assert (rewards.y == i).all()
    assert i == 99

    assert set(env.get_online_performance().keys()) == set(range(100))
    for i, (step, metric) in enumerate(env.get_online_performance().items()):
        assert step == i
        assert metric.accuracy == (1.0 if (i % 100) < 50 else 0.0), (i, step, metric)

    metrics = env.get_average_online_performance()
    assert isinstance(metrics, ClassificationMetrics)
    # Since we guessed the correct class only during the first 50 steps.
    assert metrics.accuracy == 0.5
    assert metrics.n_samples == 100


def test_measure_performance_wrapper_odd_vs_even():
    env = make_dummy_env(n_samples=100, batch_size=1)
    env = MeasureSLPerformanceWrapper(env, first_epoch_only=True)

    for i, (observations, rewards) in enumerate(env):
        assert observations is not None
        assert rewards is None or rewards.y is None
        assert (observations.x == i).all()

        # Only guess correctly for the first 50 steps.
        action = Actions(y_pred=np.array([i if i % 2 == 0 else 0]))
        rewards = env.send(action)
        assert (rewards.y == i).all()
    assert i == 99

    assert set(env.get_online_performance().keys()) == set(range(100))
    for i, (step, metric) in enumerate(env.get_online_performance().items()):
        assert step == i
        if step % 2 == 0:
            assert metric.accuracy == 1.0, (i, step, metric)
        else:
            assert metric.accuracy == 0.0, (i, step, metric)

    metrics = env.get_average_online_performance()
    assert isinstance(metrics, ClassificationMetrics)
    # Since we guessed the correct class only during the first 50 steps.
    assert metrics.accuracy == 0.5
    assert metrics.n_samples == 100


def test_measure_performance_wrapper_odd_vs_even_passive():
    dataset = TensorDataset(
        torch.arange(100).reshape([100, 1, 1, 1]) * torch.ones([100, 3, 32, 32]),
        torch.arange(100),
    )
    pretend_to_be_active = False
    env = PassiveEnvironment(
        dataset, batch_size=1, n_classes=100, pretend_to_be_active=pretend_to_be_active
    )
    env = TypedObjectsWrapper(
        env, observations_type=Observations, actions_type=Actions, rewards_type=Rewards
    )
    env = MeasureSLPerformanceWrapper(env, first_epoch_only=False)

    for i, (observations, rewards) in enumerate(env):
        assert observations is not None
        assert rewards is None or rewards.y is None
        assert (observations.x == i).all()

        # Only guess correctly for the first 50 steps.
        action = Actions(y_pred=np.array([i if i % 2 == 0 else 0]))
        rewards = env.send(action)
        assert (rewards.y == i).all()
    assert i == 99

    assert set(env.get_online_performance().keys()) == set(range(100))
    for i, (step, metric) in enumerate(env.get_online_performance().items()):
        assert step == i
        if step % 2 == 0:
            assert metric.accuracy == 1.0, (i, step, metric)
        else:
            assert metric.accuracy == 0.0, (i, step, metric)

    metrics = env.get_average_online_performance()
    assert isinstance(metrics, ClassificationMetrics)
    # Since we guessed the correct class only during the first 50 steps.
    assert metrics.accuracy == 0.5
    assert metrics.n_samples == 100


def test_last_batch():
    """ Test what happens with the last batch, in the case where the batch size doesn't
    divide the dataset equally.
    """
    env = make_dummy_env(n_samples=110, batch_size=20)
    env = MeasureSLPerformanceWrapper(env, first_epoch_only=True)

    for i, (obs, rew) in enumerate(env):
        assert rew is None
        if i != 5:
            assert obs.batch_size == 20, i
        else:
            assert obs.batch_size == 10, i
        actions = Actions(y_pred=torch.arange(i * 20, (i + 1) * 20)[: obs.batch_size])
        rewards = env.send(actions)
        assert (rewards.y == torch.arange(i * 20, (i + 1) * 20)[: obs.batch_size]).all()

    perf = env.get_average_online_performance()
    assert perf.accuracy == 1.0
    assert perf.n_samples == 110


from sequoia.methods.models.baseline_model import BaselineModel


def test_last_batch_baseline_model():
    """ BUG: Baseline method is doing something weird at the last batch, and I dont know quite why.
    """
    n_samples = 110
    batch_size = 20

    # Note: the y's here are different.
    dataset = TensorDataset(
        torch.arange(n_samples).reshape([n_samples, 1, 1, 1])
        * torch.ones([n_samples, 3, 32, 32]),
        torch.zeros(n_samples, dtype=int),
    )
    pretend_to_be_active = False
    env = PassiveEnvironment(
        dataset,
        batch_size=batch_size,
        n_classes=n_samples,
        pretend_to_be_active=pretend_to_be_active,
    )
    env = TypedObjectsWrapper(
        env, observations_type=Observations, actions_type=Actions, rewards_type=Rewards
    )
    env = MeasureSLPerformanceWrapper(env, first_epoch_only=True)
    setting = ClassIncrementalSetting()
    setting.train_env = env
    model = BaselineModel(
        setting=setting, hparams=BaselineModel.HParams(), config=Config(debug=True)
    )

    for i, (obs, rew) in enumerate(env):
        # assert rew is None
        # if i != 5:
        #     assert obs.batch_size == 20, i
        # else:
        #     assert obs.batch_size == 10, i
        # actions = Actions(y_pred=torch.arange(i * 20 , (i+1) * 20)[:obs.batch_size])
        # rewards = env.send(actions)
        # assert (rewards.y == torch.arange(i * 20 , (i+1) * 20)[:obs.batch_size]).all()
        obs = dataclasses.replace(
            obs, task_labels=torch.ones([obs.x.shape[0]], device=obs.x.device)
        )
        assert rew is None
        stuff = model.training_step((obs, rew), batch_idx=i)
        print(stuff)

    perf = env.get_average_online_performance()
    assert perf.n_samples == 110


def test_delayed_actions():
    """ Test that whenever some intermediate between the env and the Method is
    caching some of the observations, the actions and rewards still end up lining up.
    
    This is just to replicate what's happening in Pytorch Lightning, where they use some
    function to check if the batch is the last one or not, and was causing issue before.
    """
    env = make_dummy_env(n_samples=110, batch_size=20)
    env = MeasureSLPerformanceWrapper(env, first_epoch_only=True)

    T = TypeVar("T")

    def with_is_last(iterable: Iterable[T]) -> Iterable[Tuple[T, bool]]:
        iterator = iter(iterable)
        sentinel = object()
        previous_value = next(iterator)
        current_value = next(iterator, sentinel)
        while current_value is not sentinel:
            yield previous_value, False
            previous_value = current_value
            current_value = next(iterator, sentinel)
        yield previous_value, True

    for i, ((obs, rew), is_last) in enumerate(with_is_last(env)):
        print(i)
        assert rew is None
        if i != 5:
            assert obs.batch_size == 20, i
        else:
            assert obs.batch_size == 10, i
        actions = Actions(y_pred=torch.arange(i * 20, (i + 1) * 20)[: obs.batch_size])
        rewards = env.send(actions)
        assert (rewards.y == torch.arange(i * 20, (i + 1) * 20)[: obs.batch_size]).all()

    perf = env.get_average_online_performance()
    assert perf.accuracy == 1.0
    assert perf.n_samples == 110


from sequoia.common.metrics.rl_metrics import EpisodeMetrics
from sequoia.conftest import DummyEnvironment

from .measure_performance_wrapper import MeasureRLPerformanceWrapper
from itertools import accumulate
from sequoia.settings.rl.continual import ContinualRLSetting


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
