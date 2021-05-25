from typing import ClassVar, Dict, Type

import pytest
from sequoia.common.config import Config
from sequoia.settings.rl import DiscreteTaskAgnosticRLSetting

from .base import BaseAlgorithm, StableBaselines3Method
from .base_test import DiscreteActionSpaceMethodTests
from .off_policy_method import OffPolicyAlgorithm, OffPolicyMethod


class OffPolicyMethodTests:
    Method: ClassVar[Type[OffPolicyMethod]]
    Model: ClassVar[Type[OffPolicyAlgorithm]]
    debug_dataset: ClassVar[str]
    debug_kwargs: ClassVar[Dict] = {}
    
    @pytest.mark.parametrize("clear_buffers", [False, True])
    def test_clear_buffers_between_tasks(self, clear_buffers: bool, config: Config):
        setting = DiscreteTaskAgnosticRLSetting(
            dataset=self.debug_dataset,
            nb_tasks=2,
            steps_per_task=1_000,
            test_steps_per_task=1_000,
            config=config,
        )
        setting.setup()

        method = self.Method(hparams=self.Model.HParams(clear_buffers_between_tasks=clear_buffers))
        method.configure(setting)
        method.fit(
            train_env=setting.train_dataloader(), valid_env=setting.val_dataloader(),
        )
        # TODO: Not clear how to check the length of the replay buffer!
        assert False, method.model.replay_buffer.pos
        
        
        method.on_task_switch(task_id=1)
        assert len(method.model.replay_buffer) == 0
        
        # raise NotImplementedError(
        #     "TODO: Check that the method clears the replay buffer after each task when `clear_buffers_between_tasks` is set to True."
        # )

