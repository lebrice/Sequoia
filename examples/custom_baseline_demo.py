""" Demo of creating a method based on the BaselineMethod. """

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Type, ClassVar

import torch
from simple_parsing import ArgumentParser

from common.loss import Loss
from common.config import Config, TrainerConfig
from methods import BaselineMethod
from methods.models import BaselineModel
from settings import ClassIncrementalSetting, Method, Results, Setting, PassiveEnvironment

from .demo_utils import make_result_dataframe
from methods.aux_tasks import EWCTask, SimCLRTask


class MyFancyEWCVariant(EWCTask):
    name: ClassVar[str] = "fancy_ewc_variant"

    def get_loss(self, *args, **kwargs) -> Loss:
        # You could do something fancy, but this is just an example.
        ewc_loss = super().get_loss(*args, **kwargs)

        fancy_ewc_loss = ewc_loss * torch.rand(1)
        
        self.model.log("fancy_ewc_loss", fancy_ewc_loss.loss)
        return fancy_ewc_loss


class CustomizedBaselineModel(BaselineModel):
    def __init__(self, setting: Setting, hparams: BaselineModel.HParams, config: Config):
        super().__init__(setting=setting, hparams=hparams, config=config)
        self.add_auxiliary_task(MyFancyEWCVariant(coefficient=1.))
        self.add_auxiliary_task(SimCLRTask(coefficient=1.))
        
        self.replay_buffer: List = []
        # (...)


class CustomMethod(BaselineMethod, target_setting=ClassIncrementalSetting):    
    def __init__(self,
                 hparams: BaselineModel.HParams,
                 config: Config = None,
                 trainer_options: TrainerConfig = None):
        super().__init__(
            hparams=hparams,
            config=config,
            trainer_options=trainer_options,
        )

    def create_model(self, setting: ClassIncrementalSetting) -> CustomizedBaselineModel:
        return CustomizedBaselineModel(setting=setting, hparams=self.hparams, config=self.config)

    def configure(self, setting: ClassIncrementalSetting):
        super().configure(setting)
    
    def fit(self, train_env=None, valid_env=None, datamodule=None):
        return super().fit(train_env=train_env, valid_env=valid_env, datamodule=datamodule)


from .quick_demo import evaluate_on_all_settings
from .quick_demo_ewc import compare_results


def demo():
    # baseline_method = BaselineMethod.from_args()
    # baseline_results = evaluate_on_all_settings(baseline_method, below=ClassIncrementalSetting)
    
    customized_baseline_method = create_method()
    customized_baseline_results = evaluate_on_all_settings(customized_baseline_method)

    # compare_results({
    #     BaselineMethod: baseline_results,
    #     ImprovedDemoMethod: customized_baseline_results,
    # })


def create_method() -> CustomMethod:
    from simple_parsing import ArgumentParser
    
    # Get the hparams and configuration options from the command-line.
    parser = ArgumentParser(description=__doc__)
    
    parser.add_arguments(CustomizedBaselineModel.HParams, dest="hparams")
    parser.add_arguments(TrainerConfig, dest="trainer_options")
    parser.add_arguments(Config, dest="config")
    
    args = parser.parse_args()
    
    hparams: CustomizedBaselineModel.HParams = args.hparams
    trainer_options: TrainerConfig = args.trainer_options
    config: Config = args.config
    
    method = CustomMethod(
        hparams=hparams,
        trainer_options=trainer_options,
        config=config,
    )
    return method



if __name__ == "__main__":
    demo()