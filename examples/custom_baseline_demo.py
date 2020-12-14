""" Demo of creating a new Method, this time based on the BaselineMethod.

Also demonstrates how to add self-supervised "Auxiliary Tasks" the BaselineModel
as well as give an example of customizing those.
"""

import random
import sys
from dataclasses import dataclass
from typing import ClassVar, List

from simple_parsing import ArgumentParser

# This "hack" is required so we can run `python examples/custom_baseline_demo.py`
sys.path.extend([".", ".."])

from sequoia.common.config import Config, TrainerConfig
from sequoia.common.loss import Loss
from sequoia.methods import BaselineMethod
from sequoia.methods.aux_tasks import EWCTask, SimCLRTask
from sequoia.methods.models import BaselineModel
from sequoia.settings import ClassIncrementalSetting, Setting

from examples.demo_utils import compare_results, demo_all_settings


class MyFancyEWCVariant(EWCTask):
    """ Example of how to create a variant of an existing "Auxiliary task", in
    this case EWC.
    """
    name: ClassVar[str] = "fancy_ewc_variant"

    def get_loss(self, *args, **kwargs) -> Loss:
        # You could do something fancy, but this is just an example.
        ewc_loss = super().get_loss(*args, **kwargs)

        fancy_ewc_loss = ewc_loss * random.random()
        
        self.model.log("fancy_ewc_loss", fancy_ewc_loss.loss)
        return fancy_ewc_loss


class CustomizedBaselineModel(BaselineModel):
    def __init__(self, setting: Setting, hparams: BaselineModel.HParams, config: Config):
        super().__init__(setting=setting, hparams=hparams, config=config)
        
        # For example, adds the SimCLR auxiliary class, as well as the EWC aux
        # task.
        self.add_auxiliary_task(MyFancyEWCVariant(coefficient=1.))
        self.add_auxiliary_task(SimCLRTask(coefficient=1.))
        
        self.replay_buffer: List = []
        # (...)

@dataclass
class CustomMethod(BaselineMethod, target_setting=ClassIncrementalSetting):
    def __init__(self,
                 hparams: CustomizedBaselineModel.HParams = None,
                 config: Config = None,
                 trainer_options: TrainerConfig = None,
                 **kwargs):
        super().__init__(
            hparams=hparams,
            config=config,
            trainer_options=trainer_options,
            **kwargs,
        )

    def create_model(self, setting: ClassIncrementalSetting) -> CustomizedBaselineModel:
        return CustomizedBaselineModel(setting=setting, hparams=self.hparams, config=self.config)

    def configure(self, setting: ClassIncrementalSetting):
        super().configure(setting)
    
    def fit(self, train_env=None, valid_env=None, datamodule=None):
        return super().fit(train_env=train_env, valid_env=valid_env, datamodule=datamodule)


if __name__ == "__main__":
    from sequoia.settings import ClassIncrementalSetting, TaskIncrementalSetting
    from simple_parsing import ArgumentParser

    ## 1. Create the Setting from the command-line.
    parser = ArgumentParser(description=__doc__)
    parser.add_arguments(ClassIncrementalSetting, dest="setting")
    
    ## NOTE: We could also create our new method from the command-line, but this
    ## adds a lot of arguments, so we'll create it manually instead for now.
    # parser.add_arguments(CustomMethod, "method")
    
    args = parser.parse_args()
    
    setting: ClassIncrementalSetting = args.setting
    # method: CustomMethod = args.method
    
    # Here we create the arguments to be passed to our Method's constructor.
    hparams = BaselineModel.HParams()
    config = Config(debug=True)
    trainer_options = TrainerConfig(max_epochs=1)
    
    # Get the results for the base method:
    base_method = BaselineMethod(hparams=hparams, config=config, trainer_options=trainer_options)
    base_results = setting.apply(base_method)
    
    # Get the results for the 'improved' method:
    new_method = CustomMethod(hparams=hparams, config=config, trainer_options=trainer_options)
    new_results = setting.apply(new_method)
    
    print(f"\n\nComparison: BaselineMethod vs CustomMethod - (TaskIncrementalSetting, dataset=fashionmnist):")
    print(base_results.summary())
    print(new_results.summary())
    
    exit()
    
    # Optionally, second part of the demo:
    # baseline_method = BaselineMethod.from_args()
    baseline_results = demo_all_settings(BaselineMethod, below=ClassIncrementalSetting)
    
    # # Since we inherit from the BaselineMethod, we can just call 'from_args':
    customized_baseline_results = demo_all_settings(CustomMethod, below=ClassIncrementalSetting)

    compare_results({
        BaselineMethod: baseline_results,
        CustomMethod: customized_baseline_results,
    })    
    

