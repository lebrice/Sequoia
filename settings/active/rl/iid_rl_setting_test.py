from dataclasses import dataclass
from typing import ClassVar, Dict

from common.config import Config
from conftest import DummyEnvironment
from methods import RandomBaselineMethod
from utils import take

from .iid_rl_setting import RLSetting

# TODO: Write some tests to make sure that the actions actually get sent back
# to the loaders for each of 'train' 'val' and 'test'. 


def test_basic():
    batch_size: int = 10
    expected_obs_shape = (4,)
    dataset: str = "cartpole"
    observe_state_directly = True

    setting = RLSetting(observe_state_directly=observe_state_directly, dataset=dataset)
    setting.prepare_data()
    setting.setup()

    expected_obs_batch_shape = (batch_size, *expected_obs_shape)
    # Test the shapes of the obs generated by the train/val/test dataloaders.
    dataloader_methods = [
        setting.train_dataloader,
        setting.val_dataloader,
        setting.test_dataloader
    ]
    for dataloader_method in dataloader_methods:
        dataloader = dataloader_method(batch_size=batch_size)
        reset_obs = dataloader.reset()
        assert reset_obs.shape == expected_obs_batch_shape
        step_obs, *_ = dataloader.step(dataloader.random_actions())
        assert step_obs.shape == expected_obs_batch_shape
        for iter_obs, *_ in take(dataloader, 3):
            assert iter_obs.shape == expected_obs_batch_shape 
            reward = dataloader.send(dataloader.random_actions())