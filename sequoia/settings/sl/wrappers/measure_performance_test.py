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
from sequoia.settings.rl.wrappers import TypedObjectsWrapper
from sequoia.settings.rl.continual.setting import ContinualRLSetting
from sequoia.settings.sl.environment import PassiveEnvironment
from sequoia.settings.sl import ClassIncrementalSetting
from .measure_performance import MeasureSLPerformanceWrapper
from sequoia.settings.sl.incremental.objects import Actions, Observations, Rewards


T = TypeVar("T")


def with_is_last(iterable: Iterable[T]) -> Iterable[Tuple[T, bool]]:
    """ Function that mimics what's happening in pytorch-lightning, where the iterator
    is one-offset. This can cause a bit of headache in Sequoia's wrappers when iterating
    over an env, because they expect an action for each observation.
    """
    iterator = iter(iterable)
    sentinel = object()
    previous_value = next(iterator)
    current_value = next(iterator, sentinel)
    while current_value is not sentinel:
        yield previous_value, False
        previous_value = current_value
        current_value = next(iterator, sentinel)
    yield previous_value, True


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


def make_dummy_env(n_samples: int = 100, batch_size: int = 1, drop_last: bool = False):
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
        drop_last=drop_last,
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


from sequoia.methods.models.base_model import BaseModel


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
    model = BaseModel(
        setting=setting, hparams=BaseModel.HParams(), config=Config(debug=True)
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


@pytest.mark.parametrize("drop_last", [False, True])
def test_delayed_actions(drop_last: bool):
    """ Test that whenever some intermediate between the env and the Method is
    caching some of the observations, the actions and rewards still end up lining up.
    
    This is just to replicate what's happening in Pytorch Lightning, where they use some
    function to check if the batch is the last one or not, and was causing issue before.
    """
    env = make_dummy_env(n_samples=110, batch_size=20, drop_last=drop_last)
    env = MeasureSLPerformanceWrapper(env, first_epoch_only=True)
    i = 0
    
    for i, ((obs, rew), is_last) in enumerate(with_is_last(env)):
        print(i, obs.batch_size)
        assert rew is None
        if i != 5:
            assert obs.batch_size == 20, i
        else:
            assert obs.batch_size == 10, i
        actions = Actions(y_pred=torch.arange(i * 20, (i + 1) * 20)[: obs.batch_size])
        rewards = env.send(actions)
        assert (rewards.y == torch.arange(i * 20, (i + 1) * 20)[: obs.batch_size]).all()    
    assert i == (4 if drop_last else 5)
    assert is_last
    
    for i, ((obs, rew), is_last) in enumerate(with_is_last(env)):
        print(i)
        # We get rewards now that we're outside of the first epoch.
        assert rew is not None
        if i < 5:
            assert obs.batch_size == 20, i
        else:
            assert obs.batch_size == 10, i

        # actions = Actions(y_pred=torch.arange(i * 20, (i + 1) * 20)[: obs.batch_size])
        # rewards = env.send(actions)
        # assert (rewards.y == torch.arange(i * 20, (i + 1) * 20)[: obs.batch_size]).all()    
    assert i == 4 if drop_last else 5
    assert len(list(env)) == 5 if drop_last else 6
    assert len(list(with_is_last(env))) == 5 if drop_last else 6
    

    perf = env.get_average_online_performance()
    assert perf.accuracy == 1.0
    # BUG: The number of samples for the metrics isn't quite right, should include the
    # last batch, even if it doesn't have a 'full' batch.
    assert perf.n_samples == (100 if drop_last else 110)

