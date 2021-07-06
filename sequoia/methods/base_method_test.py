from typing import Type, ClassVar, Dict

import numpy as np
import pytest
import torch
from pytorch_lightning import Trainer
from sequoia.common.config import Config, TrainerConfig
from sequoia.conftest import skip_param, slow
from sequoia.settings import (
    ClassIncrementalSetting,
    IncrementalRLSetting,
    TraditionalRLSetting,
    Setting,
)
from sequoia.settings.rl.continual.results import ContinualRLResults

from .base_method import BaseMethod, BaseModel
from .method_test import MethodTests


@pytest.fixture(scope="module")
def trainer_options(tmp_path_factory) -> TrainerConfig:
    tmp_path = tmp_path_factory.mktemp("log_dir")
    return TrainerConfig(
        # logger=False,
        max_epochs=1,
        checkpoint_callback=False,
        default_root_dir=tmp_path,
    )


@pytest.fixture(scope="module")
def trainer(trainer_options: TrainerConfig, config: Config):
    """ Fixture that produces a pl.Trainer to be used during testing. """
    # TODO: Parametrize with the accelerator to use, skip param if no GPU?
    return trainer_options.make_trainer(config=config, loggers=None)


class TestBaseMethod(MethodTests):
    Method: ClassVar[Type[BaseMethod]] = BaseMethod
    method_debug_kwargs: Dict = {"max_epochs": 1}

    @classmethod
    @pytest.fixture
    def method(cls, config: Config, trainer_options: TrainerConfig) -> BaseMethod:
        """ Fixture that returns the Method instance to use when testing/debugging.
        """
        trainer_options.max_epochs = 1
        return cls.Method(trainer_options=trainer_options, config=config)

    @slow
    @pytest.mark.timeout(120)
    def test_cartpole_state(self, config: Config, trainer_options: TrainerConfig):
        """ Test that the baseline method can learn cartpole (state input) """
        # TODO: Actually remove the trainer_config class from the BaseMethod?
        trainer_options.max_epochs = 1
        method = self.Method(config=config, trainer_options=trainer_options)
        method.hparams.learning_rate = 0.01

        setting = TraditionalRLSetting(
            dataset="CartPole-v0",
            train_max_steps=5000,
            nb_tasks=1,
            test_max_steps=2_000,
            config=config,
        )
        results: ContinualRLResults = setting.apply(method)

        print(results.to_log_dict())
        # The method should normally get the maximum length (200), but checking with
        # 100 just to account for randomness.
        assert results.average_metrics.mean_episode_length > 100.0

    @slow
    @pytest.mark.timeout(120)
    def test_incremental_cartpole_state(
        self, config: Config, trainer_options: TrainerConfig
    ):
        """ Test that the baseline method can learn cartpole (state input) """
        # TODO: Actually remove the trainer_config class from the BaseMethod?
        trainer_options.max_epochs = 1
        method = self.Method(config=config, trainer_options=trainer_options)
        method.hparams.learning_rate = 0.01

        setting = IncrementalRLSetting(
            dataset="cartpole", train_max_steps=5000, nb_tasks=2, test_max_steps=1000
        )
        results: ContinualRLResults = setting.apply(method)

        print(results.to_log_dict())
        # The method should normally get the maximum length (200), but checking with
        # 100 just to account for randomness.
        assert results.mean_episode_length > 100.0

    @pytest.mark.timeout(30)
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Cuda is required.")
    def test_device_of_output_head_is_correct(
        self,
        short_class_incremental_setting: ClassIncrementalSetting,
        trainer_options: TrainerConfig,
        config: Config,
    ):
        """ There is a bug happening where the output head is on CPU while the rest of the
        model is on GPU.
        """
        trainer_options.max_epochs = 1
        method = self.Method(trainer_options=trainer_options, config=config)
        results = short_class_incremental_setting.apply(method)
        assert 0.20 <= results.objective


BaseMethodTests = TestBaseMethod
