import numpy as np
import pytest
import torch
from gym import spaces
from sequoia.common.config import Config
from sequoia.common.spaces import Image, NamedTupleSpace, Sparse
from sequoia.conftest import monsterkong_required
from sequoia.settings.active import (
    ContinualRLSetting,
    IncrementalRLSetting,
    TaskIncrementalRLSetting,
)

from .dqn import DQNMethod, DQNModel


def test_cartpole_state():
    method = DQNMethod(hparams=DQNModel.HParams(train_freq=1))
    setting = IncrementalRLSetting(
        dataset="cartpole",
        observe_state_directly=True,
        nb_tasks=2,
        steps_per_task=1_000,
        test_steps_per_task=1_000,
    )
    results: IncrementalRLSetting.Results = setting.apply(
        method, config=Config(debug=True)
    )
    print(results.summary())


@pytest.mark.timeout(60)
@monsterkong_required
def test_monsterkong():
    method = DQNMethod(hparams=DQNModel.HParams(train_freq=1))
    setting = IncrementalRLSetting(
        dataset="monsterkong",
        nb_tasks=2,
        steps_per_task=1_000,
        test_steps_per_task=1_000,
    )
    results: IncrementalRLSetting.Results = setting.apply(
        method, config=Config(debug=True)
    )
    print(results.summary())


def test_dqn_monsterkong_adds_channel_first_transform():
    method = DQNMethod()
    setting = IncrementalRLSetting(
        dataset="monsterkong",
        nb_tasks=2,
        steps_per_task=1_000,
        test_steps_per_task=1_000,
    )
    assert setting.max_steps == 2_000
    assert setting.test_steps == 2_000
    assert setting.nb_tasks == 2
    assert setting.observation_space == NamedTupleSpace(
        spaces={
            "x": Image(0, 1, shape=(3, 64, 64), dtype=np.float32),
            "task_labels": Sparse(spaces.Discrete(2)),
        },
        dtype=setting.Observations,
    )
    assert setting.action_space == spaces.Discrete(6)  # monsterkong has 6 actions.

    # (Before the method gets to change the Setting):
    # By default the setting gives the same shape of obs as the underlying env.
    for env_method in [
        setting.train_dataloader,
        setting.val_dataloader,
        setting.test_dataloader,
    ]:
        print(f"Testing method {env_method.__name__}")
        with env_method() as env:
            reset_obs = env.reset()
            # TODO: Fix this so the 'x' space actually gets tensor support.
            # assert reset_obs in env.observation_space
            assert reset_obs.numpy() in env.observation_space
            assert reset_obs.x.shape == (3, 64, 64)

    # Let the Method configure itself on the Setting:
    method.configure(setting)

    # (After the method gets to change the Setting):

    for env_method in [
        setting.train_dataloader,
        setting.val_dataloader,
        setting.test_dataloader,
    ]:
        with env_method() as env:
            reset_obs = env.reset()
            # Fix this numpy bug.
            assert reset_obs.numpy() in env.observation_space
            assert reset_obs.x.shape == (64, 64, 3)
