from typing import Callable, List, Optional, Tuple

import gym
import numpy as np
import pytest
from gym import spaces

from sequoia.common.config import Config
from sequoia.common.spaces import Image, Sparse
from sequoia.common.transforms import (ChannelsFirstIfNeeded, ToTensor,
                                       Transforms)
from sequoia.conftest import xfail_param, monsterkong_required
from sequoia.settings import Method
from sequoia.settings.assumptions.incremental import TestEnvironment
from sequoia.utils.utils import take

from .incremental_rl_setting import IncrementalRLSetting


@pytest.mark.parametrize("batch_size", [None, 1, 3])
@pytest.mark.parametrize(
    "dataset, expected_obs_shape", [
        ("CartPole-v0", (3, 400, 600)),
        # ("Breakout-v0", (3, 210, 160)),
        ("Breakout-v0", (3, 84, 84)), # Since the Atari Preprocessing is added by default.
        # ("duckietown", (120, 160, 3)),
    ],
)
def test_check_iterate_and_step(dataset: str,
                                expected_obs_shape: Tuple[int, ...],
                                batch_size: int):
    setting = IncrementalRLSetting(dataset=dataset, nb_tasks=5)
    assert len(setting.train_task_schedule) == 5
    assert not setting.smooth_task_boundaries
    assert setting.task_labels_at_train_time
    
    # TODO: Should we have the task label space in this case?
    assert setting.task_labels_at_train_time
    assert not setting.task_labels_at_test_time
    
    if batch_size is None:
        expected_obs_batch_shape = expected_obs_shape
    else:
        expected_obs_batch_shape = (batch_size, *expected_obs_shape)
    
    with setting.train_dataloader(batch_size=batch_size) as temp_env:
        obs_space = temp_env.observation_space
        assert obs_space[0] == spaces.Box(0., 1., expected_obs_batch_shape, dtype=np.float32)
        assert obs_space[1] == spaces.MultiDiscrete([5] * batch_size) if batch_size else spaces.Discrete(5)

    with setting.val_dataloader(batch_size=batch_size) as temp_env:
        # No task labels:
        obs_space = temp_env.observation_space

        assert obs_space[0] == spaces.Box(0., 1., expected_obs_batch_shape, dtype=np.float32)
        if batch_size:
            assert str(obs_space[1]) == str(spaces.MultiDiscrete([5] * batch_size))
            # assert str(obs_space[1]) == str(spaces.Tuple([Sparse(spaces.Discrete(5), sparsity=1.) for _ in range(batch_size)]))
        else:
            # TODO: Should the task labels be given in the valid dataloader if they arent' during testing?
            assert obs_space[1] == spaces.Discrete(5)
            # assert obs_space[1] == Sparse(spaces.Discrete(5), sparsity=1.)

    # NOTE: Limitting the batch size at test time to None (i.e. a single env)
    # because of how the Monitor class works atm.
    
    with setting.test_dataloader(batch_size=None) as temp_env:
        obs_space = temp_env.observation_space
        assert obs_space[1] == Sparse(spaces.Discrete(5), sparsity=1.)
        # No task labels:
        # if batch_size:
        #     assert str(obs_space[1]) == str(spaces.Tuple([Sparse(spaces.Discrete(5), sparsity=1.) for _ in range(batch_size)]))

    def check_obs(obs, task_label: int = None):
        if batch_size is None:
            assert obs[1] == task_label
        else:
            assert isinstance(obs, IncrementalRLSetting.Observations), obs[0].shape
            assert obs.task_labels is task_label or all(task_label == task_label for task_label in obs.task_labels)

    env = setting.train_dataloader(batch_size=batch_size)
    reset_obs = env.reset()
    check_obs(reset_obs, task_label=0)
    
    for i in range(5):
        step_obs, *_ = env.step(env.action_space.sample())
        check_obs(step_obs, task_label=0)

    for iter_obs in take(env, 3):
        check_obs(iter_obs, task_label=0)
        reward = env.send(env.action_space.sample())
        env.render("human")
        
    env.close()




class DummyMethod(Method, target_setting=IncrementalRLSetting):
    """ Dummy method used to check that the Setting calls `on_task_switch` with the
    right arguments. 
    """
    def __init__(self):
        self.n_task_switches = 0
        self.received_task_ids: List[Optional[int]] = []
    
    def fit(self, train_env: gym.Env = None, valid_env: gym.Env = None):
        obs = train_env.reset()
        for i in range(100):
            obs, reward, done, info = train_env.step(train_env.action_space.sample())
            if done:
                break
    
    def test(self, test_env: TestEnvironment):
        
        while not test_env.is_closed():
            done = False
            obs = test_env.reset()
            while not done:
                actions = test_env.action_space.sample()
                obs, _, done, info = test_env.step(actions)

    def get_actions(self, observations: IncrementalRLSetting.Observations, action_space: gym.Space):
        return np.ones(action_space.shape)

    def on_task_switch(self, task_id: int=None):
        self.n_task_switches += 1
        self.received_task_ids.append(task_id)


from sequoia.conftest import DummyEnvironment


def test_on_task_switch_is_called():
    setting = IncrementalRLSetting(
        dataset=DummyEnvironment,
        nb_tasks=5,
        steps_per_task=100,
        max_steps=500,
        train_transforms=[],
        test_transforms=[],
        val_transforms=[],
    )
    method = DummyMethod()
    results = setting.apply(method)
    # 5 during training, 5 during testing!
    assert method.n_task_switches == 10
    assert method.received_task_ids == list(range(5)) + [None for _ in range(5)]
    

    setting = IncrementalRLSetting(
        dataset=DummyEnvironment,
        nb_tasks=5,
        steps_per_task=100,
        max_steps=500,
        train_transforms=[],
        test_transforms=[],
        val_transforms=[],
        task_labels_at_test_time=True,
    )
    method = DummyMethod()
    results = setting.apply(method)
    # 5 during training, 5 during testing!
    assert method.n_task_switches == 10
    assert method.received_task_ids == list(range(5)) + list(range(5))

@monsterkong_required
@pytest.mark.parametrize("task_labels_at_test_time", [False, True])
def test_monsterkong_state(task_labels_at_test_time: bool):
    """ checks that the MonsterKong env works fine with monsterkong and state input. """
    setting = IncrementalRLSetting(
        dataset="monsterkong",
        observe_state_directly=True,
        nb_tasks=5,
        steps_per_task=1000,
        train_transforms=[],
        test_transforms=[],
        val_transforms=[],
        task_labels_at_test_time=task_labels_at_test_time,
        max_episode_steps=100,
    )
    with setting.train_dataloader() as env:
        obs = env.reset()
        assert obs in setting.observation_space
    
    method = DummyMethod()
    results = setting.apply(method)

    expected_test_time_task_ids = [i if task_labels_at_test_time else None for i in range(5)]
    # 5 during training, 5 during testing!
    assert method.n_task_switches == 10
    assert method.received_task_ids == list(range(5)) + expected_test_time_task_ids


@pytest.mark.timeout(120)
@monsterkong_required
@pytest.mark.parametrize("task_labels_at_test_time", [False, True])
def test_monsterkong_pixels(task_labels_at_test_time: bool):
    """ checks that the MonsterKong env works fine with monsterkong and state input. """
    setting = IncrementalRLSetting(
        dataset="monsterkong",
        observe_state_directly=False,
        nb_tasks=5,
        steps_per_task=1000,
        test_steps_per_task=1000,
        train_transforms=[],
        test_transforms=[],
        val_transforms=[],
        task_labels_at_test_time=task_labels_at_test_time,
        max_episode_steps=100,
    )
    assert setting.observation_space.x == Image(0, 255, (64, 64, 3), np.uint8)
    with setting.train_dataloader() as env:
        obs = env.reset()
        assert obs in setting.observation_space
    
    method = DummyMethod()
    results = setting.apply(method)

    expected_test_time_task_ids = [i if task_labels_at_test_time else None for i in range(5)]
    # 5 during training, 5 during testing!
    assert method.n_task_switches == 10
    assert method.received_task_ids == list(range(5)) + expected_test_time_task_ids
