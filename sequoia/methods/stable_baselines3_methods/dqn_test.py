import numpy as np
import pytest
from gym import spaces
from sequoia.common.config import Config
from sequoia.common.spaces import Image, NamedTupleSpace, Sparse
from sequoia.conftest import monsterkong_required
from sequoia.settings.rl import IncrementalRLSetting

from .dqn import DQNMethod, DQNModel
from typing import ClassVar, Type, Dict

from .base import StableBaselines3Method, BaseAlgorithm
from .base_test import DiscreteActionSpaceMethodTests
from .off_policy_method_test import OffPolicyMethodTests


class TestDQN(DiscreteActionSpaceMethodTests, OffPolicyMethodTests):
    Method: ClassVar[Type[StableBaselines3Method]] = DQNMethod
    Model: ClassVar[Type[BaseAlgorithm]] = DQNModel
    debug_kwargs: ClassVar[Dict] = {}

    # TODO: Maybe this is because of the buffer isn't filled up enough with the short
    # number of allowed steps?
    @pytest.mark.xfail(reason="DQN really sucks on cartpole?")
    def test_classic_control_state(self, config: Config):
        super().test_classic_control_state(config=config)

    @pytest.mark.xfail(reason="DQN really sucks on cartpole?")
    def test_incremental_classic_control_state(self, config: Config):
        super().test_incremental_classic_control_state(config=config)

    def test_dqn_monsterkong_adds_channel_first_transform(self):
        method = self.Method(**self.debug_kwargs)
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
                "x": Image(0, 255, shape=(64, 64, 3), dtype=np.uint8),
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
                assert reset_obs.x.shape == (64, 64, 3)

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
