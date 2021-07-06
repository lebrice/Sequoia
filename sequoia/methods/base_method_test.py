from typing import Type, ClassVar

import numpy as np
import pytest
import torch
from pytorch_lightning import Trainer
from sequoia.common.config import Config, TrainerConfig
from sequoia.conftest import skip_param, slow
from sequoia.settings import (ClassIncrementalSetting, IncrementalRLSetting,
                              TraditionalRLSetting, Setting)
from sequoia.settings.rl.continual.results import ContinualRLResults

from .base_method import BaseMethod, BaseModel
from .method_test import MethodTests

@pytest.fixture
def trainer(tmp_path_factory):
    """ Fixture that produces a pl.Trainer to be used during testing. """
    tmp_path = tmp_path_factory.mktemp("log_dir")
    # TODO: Parametrize with the accelerator to use, skip param if no GPU?
    return Trainer(
        logger=False,
        checkpoint_callback=False,
        default_root_dir=tmp_path,
    )


class TestBaseMethod(MethodTests):
    Method: ClassVar[Type[BaseMethod]] = BaseMethod

    @slow
    @pytest.mark.timeout(120)
    def test_cartpole_state(self, config: Config, trainer: Trainer):
        """ Test that the baseline method can learn cartpole (state input) """
        # TODO: Actually remove the trainer_config class from the BaseMethod?
        method = self.Method(config=config)
        method.trainer = trainer
        method.hparams.learning_rate = 0.01

        setting = TraditionalRLSetting(dataset="CartPole-v0", train_max_steps=5000, nb_tasks=1, test_max_steps=2_000)
        results: ContinualRLResults = setting.apply(method)

        print(results.to_log_dict())
        # The method should normally get the maximum length (200), but checking with
        # 100 just to account for randomness.
        assert results.average_metrics.mean_episode_length > 100.

    @slow
    @pytest.mark.timeout(120)
    def test_incremental_cartpole_state(self, config: Config, trainer: Trainer):
        """ Test that the baseline method can learn cartpole (state input) """
        # TODO: Actually remove the trainer_config class from the BaseMethod?
        method = self.Method(config=config)
        method.trainer = trainer
        method.hparams.learning_rate = 0.01
        
        setting = IncrementalRLSetting(dataset="cartpole", train_max_steps=5000, nb_tasks=2, test_max_steps=1000)
        results: ContinualRLResults = setting.apply(method)

        print(results.to_log_dict())
        # The method should normally get the maximum length (200), but checking with
        # 100 just to account for randomness.
        assert results.mean_episode_length > 100.

    @pytest.mark.timeout(30)
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Cuda is required.")
    def test_device_of_output_head_is_correct(self, short_class_incremental_setting: ClassIncrementalSetting):
        """ There is a bug happening where the output head is on CPU while the rest of the
        model is on GPU.
        """
        method = self.Method(max_epochs=1, no_wandb=True)
        results = short_class_incremental_setting.apply(method)
        assert 0.25 <= results.objective


BaseMethodTests = TestBaseMethod
