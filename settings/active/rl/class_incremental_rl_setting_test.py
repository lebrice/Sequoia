from typing import Callable, List, Optional, Tuple

import gym
import pytest
import numpy as np
from gym import spaces

from common.gym_wrappers.sparse_space import Sparse
from common.config import Config
from common.transforms import ChannelsFirstIfNeeded, ToTensor, Transforms
from conftest import xfail_param
from utils.utils import take

from .class_incremental_rl_setting import ClassIncrementalRLSetting


@pytest.mark.parametrize("batch_size", [None, 1, 3])
@pytest.mark.parametrize(
    "dataset, expected_obs_shape", [
        ("CartPole-v0", (3, 400, 600)),
        ("Breakout-v0", (3, 210, 160)),
        # ("duckietown", (120, 160, 3)),
    ],
)
def test_check_iterate_and_step(dataset: str,
                                expected_obs_shape: Tuple[int, ...],
                                batch_size: int):
    setting = ClassIncrementalRLSetting(dataset=dataset, nb_tasks=5)
    
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
        assert temp_env.observation_space == spaces.Tuple([
            spaces.Box(0., 1., expected_obs_batch_shape, dtype=np.float32),
            (spaces.MultiDiscrete([5] * batch_size) if batch_size else spaces.Discrete(5)),
        ])

    with setting.val_dataloader(batch_size=batch_size) as temp_env:
        assert temp_env.observation_space == spaces.Box(0., 1., (batch_size, 3, 210, 160), dtype=np.float32)

    with setting.test_dataloader(batch_size=batch_size) as temp_env:
        assert temp_env.observation_space == spaces.Box(0., 1., (batch_size, 3, 210, 160), dtype=np.float32)
        
    # with setting.val_dataloader(batch_size=batch_size) as valid_env:
    #     assert str(valid_env.observation_space) == str(spaces.Tuple([
    #         spaces.Box(0., 1., (batch_size, 3, 210, 160), dtype=np.float32),
    #         Sparse(spaces.MultiDiscrete([5] * batch_size), none_prob=1.),
    #     ]))

    def check_obs(obs):
        assert isinstance(obs, ClassIncrementalRLSetting.Observations), obs[0].shape
        assert obs.x.shape == expected_obs_shape
        assert obs.task_labels is None or all(task_label is None for task_label in obs.task_labels)

    env = setting.train_dataloader(batch_size=batch_size)
    reset_obs = env.reset()
    check_obs(reset_obs)
    
    step_obs, *_ = env.step(env.action_space.sample())
    check_obs(step_obs)

    for iter_obs in take(env, 3):
        check_obs(iter_obs)
        reward = env.send(env.action_space.sample())
        env.render("human")
        
        if all(done):
            break
    env.close()
