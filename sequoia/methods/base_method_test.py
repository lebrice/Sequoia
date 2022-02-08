from typing import ClassVar, Dict, Type

import pytest
import torch

from sequoia.common.config import Config
from sequoia.conftest import slow
from sequoia.methods.trainer import TrainerConfig
from sequoia.settings import (
    ClassIncrementalSetting,
    IncrementalRLSetting,
    Setting,
    TraditionalRLSetting,
)
from sequoia.settings.rl.continual.results import ContinualRLResults

from .base_method import BaseMethod
from .method_test import MethodTests


class TestBaseMethod(MethodTests):
    Method: ClassVar[Type[BaseMethod]] = BaseMethod
    method_debug_kwargs: Dict = {"max_epochs": 1}

    @classmethod
    @pytest.fixture(scope="module")
    def trainer_options(cls, tmp_path_factory) -> TrainerConfig:
        tmp_path = tmp_path_factory.mktemp("log_dir")
        return TrainerConfig(
            # logger=False,
            max_epochs=1,
            checkpoint_callback=False,
            default_root_dir=tmp_path,
        )

    @classmethod
    @pytest.fixture
    def method(cls, config: Config, trainer_options: TrainerConfig) -> BaseMethod:
        """Fixture that returns the Method instance to use when testing/debugging."""
        trainer_options.max_epochs = 1
        return cls.Method(trainer_options=trainer_options, config=config)

    def validate_results(
        self,
        setting: Setting,
        method: BaseMethod,
        results: Setting.Results,
    ) -> None:
        assert results
        assert results.objective
        # TODO: Set some 'reasonable' bounds on the performance here, depending on the
        # setting/dataset.

    @slow
    @pytest.mark.timeout(120)
    def test_cartpole_state(self, config: Config, trainer_options: TrainerConfig):
        """Test that the baseline method can learn cartpole (state input)"""
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
    def test_incremental_cartpole_state(self, config: Config, trainer_options: TrainerConfig):
        """Test that the baseline method can learn cartpole (state input)"""
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
        """There is a bug happening where the output head is on CPU while the rest of the
        model is on GPU.
        """
        trainer_options.max_epochs = 1
        method = self.Method(trainer_options=trainer_options, config=config)
        results = short_class_incremental_setting.apply(method)
        assert 0.20 <= results.objective


def test_weird_pl_bug():
    replica_device = None

    def find_tensor_with_device(tensor: torch.Tensor) -> torch.Tensor:
        nonlocal replica_device
        if replica_device is None and tensor.device != torch.device("cpu"):
            replica_device = tensor.device
        return tensor

    from pytorch_lightning.utilities.apply_func import apply_to_collection

    from sequoia.settings.sl.incremental.objects import (
        IncrementalSLObservations,
        IncrementalSLRewards,
    )

    # TODO: Not quite sure why there is also a `0` in there.
    input_device = "cuda"
    inputs = (
        (
            IncrementalSLObservations(
                x=torch.rand([32, 3, 28, 28], device=input_device),
                task_labels=torch.zeros([32], device=input_device),
            ),
            IncrementalSLRewards(y=torch.randint(10, [32], device=input_device)),
        ),
        0,
    )

    # from collections.abc import Mapping, Sequence
    apply_to_collection(inputs, dtype=torch.Tensor, function=find_tensor_with_device)

    assert replica_device is not None


BaseMethodTests = TestBaseMethod
