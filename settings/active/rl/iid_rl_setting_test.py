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

from .iid_rl_setting import RLSetting

# @pytest.mark.parametrize("dataset", ["breakout"])

def test_basic(config: Config):
    setting = RLSetting(dataset="breakout", nb_tasks=5)
    batch_size = 4
    
    assert not setting.smooth_task_boundaries
    assert setting.task_labels_at_train_time
    
    # TODO: Should we have the task label space in this case?
    assert setting.task_labels_at_train_time
    assert setting.task_labels_at_test_time
    
    with setting.train_dataloader(batch_size=batch_size) as temp_env:
        assert temp_env.observation_space == spaces.Box(0., 1., (batch_size, 3, 210, 160), dtype=np.float32)

    with setting.val_dataloader(batch_size=batch_size) as temp_env:
        assert temp_env.observation_space == spaces.Box(0., 1., (batch_size, 3, 210, 160), dtype=np.float32)

    with setting.test_dataloader(batch_size=batch_size) as temp_env:
        assert temp_env.observation_space == spaces.Box(0., 1., (batch_size, 3, 210, 160), dtype=np.float32)
        
    # with setting.val_dataloader(batch_size=batch_size) as valid_env:
    #     assert str(valid_env.observation_space) == str(spaces.Tuple([
    #         spaces.Box(0., 1., (batch_size, 3, 210, 160), dtype=np.float32),
    #         Sparse(spaces.MultiDiscrete([5] * batch_size), none_prob=1.),
    #     ]))

    for task_id in range(setting.nb_tasks):
        setting.current_task_id = task_id
        
        
        env = setting.train_dataloader(batch_size=batch_size)
        
        assert env.observation_space == spaces.Box(0., 1., (batch_size, 3, 210, 160), dtype=np.float32)
        
        observations: RLSetting.Observations = env.reset()
        if setting.task_labels_at_train_time:
            assert all(observations.task_labels == task_id)
        else:
            assert all(obs_task_id is None for obs_task_id in observations.task_labels)

        for i in range(5):
            actions = env.action_space.sample()
            observations, rewards, done, info = env.step(actions)
            
            assert isinstance(observations, RLSetting.Observations)
            assert observations.x.shape == (batch_size, 3, 210, 160)
            
            if setting.task_labels_at_train_time:
                assert all(obs_task_id == task_id for obs_task_id in observations.task_labels)
            else:
                assert observations.task_labels is None or all(obs_task_id is None for obs_task_id in observations.task_labels)
                
            # TODO: Is this what we want? Could the reward or actions ever change?
            assert isinstance(rewards, RLSetting.Rewards)
            assert rewards.y.shape == (batch_size,)
            
            env.render("human")
            
            if all(done):
                break
        env.close()
