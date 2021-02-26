from .baseline_method import BaselineMethod
import pytest
from typing import Type

from sequoia.settings import Setting, RLSetting, IncrementalRLSetting
from sequoia.common.config import Config
from sequoia.conftest import skip_param, slow
import numpy as np
from pytorch_lightning import Trainer
from sequoia.settings.active import RLResults


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


@slow
@pytest.mark.timeout(120)
def test_cartpole_state(config: Config, trainer: Trainer):
    """ Test that the baseline method can learn cartpole (state input) """
    # TODO: Actually remove the trainer_config class from the BaselineMethod?
    method = BaselineMethod(config=config)
    method.trainer = trainer
    method.hparams.learning_rate = 0.01
    
    setting = RLSetting(dataset="cartpole", max_steps=5000,
                        observe_state_directly=True)
    results: RLResults = setting.apply(method)

    print(results.to_log_dict())
    # The method should normally get the maximum length (200), but checking with
    # 100 just to account for randomness.
    assert results.mean_episode_length > 100.



@slow
@pytest.mark.timeout(120)
def test_incremental_cartpole_state(config: Config, trainer: Trainer):
    """ Test that the baseline method can learn cartpole (state input) """
    # TODO: Actually remove the trainer_config class from the BaselineMethod?
    method = BaselineMethod(config=config)
    method.trainer = trainer
    method.hparams.learning_rate = 0.01
    
    setting = IncrementalRLSetting(dataset="cartpole", max_steps=5000,
                        observe_state_directly=True, nb_tasks=2)
    results: RLResults = setting.apply(method)

    print(results.to_log_dict())
    # The method should normally get the maximum length (200), but checking with
    # 100 just to account for randomness.
    assert results.mean_episode_length > 100.



from .baseline_method import BaselineModel
import torch
from sequoia.common.config import Config, TrainerConfig
import pytest
from sequoia.settings import ClassIncrementalSetting


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Cuda is required.")
def test_device_of_output_head_is_correct(config):
    """ There is a bug happening where the output head is on CPU while the rest of the
    model is on GPU.
    """ 
    setting = ClassIncrementalSetting(dataset="mnist")
    config.device = "cuda"
    method = BaselineMethod(hparams=BaselineModel.HParams(), config=config, trainer_options=TrainerConfig(fast_dev_run=True))

    results = setting.apply(method, config=config)
    assert False, results.objective