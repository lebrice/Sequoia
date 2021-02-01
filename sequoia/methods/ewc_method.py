# Imports:
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Type, Union

from nngeometry.generator.jacobian import Jacobian
from simple_parsing import ArgumentParser, choice
from torch.cuda import device

sys.path.extend([".", ".."])
from sequoia.common.config import Config
from sequoia.common.config.trainer_config import TrainerConfig
from sequoia.methods import register_method
from sequoia.methods.baseline_method import BaselineMethod, BaselineModel
# Repo imports:
from sequoia.settings import (ActiveEnvironment, ActiveSetting,
                              ClassIncrementalSetting, ContinualRLSetting,
                              IncrementalRLSetting, PassiveEnvironment,
                              PassiveSetting, RLSetting, Setting,
                              TaskIncrementalRLSetting, TaskIncrementalSetting)
from sequoia.settings.assumptions.incremental import IncrementalSetting
from simple_parsing import mutable_field

from sequoia.methods.aux_tasks import AuxiliaryTask
from sequoia.methods.aux_tasks.ewc import EWCTask


class EwcModel(BaselineModel):
    """ Modified version of the BaselineModel, which adds the EWC auxiliary task. """

    @dataclass
    class HParams(BaselineModel.HParams):
        """ Hyper-parameters of the `EwcModel`. """

        # Hyper-parameters related to the EWC auxiliary task.
        ewc: EWCTask.Options = mutable_field(EWCTask.Options, coefficient=100)

    def __init__(self, setting: Setting, hparams: "EwcModel.HParams", config: Config):
        super().__init__(setting=setting, hparams=hparams, config=config)
        self.hp: EwcModel.HParams

        self.FIM: Jacobian = None
        self.FIM_representation = self.hp.ewc.fim_representation
        self.last_task_train_env: PassiveEnvironment = None

        # self.device = device
        self.previous_model_weights = None
        self._n_switches: int = 0
        self._previous_task_id: int = 0
        # BUG: mutable_field doesn't work correctly, should use default value of 100!
        self.add_auxiliary_task(EWCTask(options=self.hp.ewc), coefficient=100)

    @property
    def ewc_aux_task(self) -> EWCTask:
        return self.tasks[EWCTask.name]
    
    def create_auxiliary_tasks(self) -> Dict[str, AuxiliaryTask]:
        tasks = super().create_auxiliary_tasks()
        # tasks["ewc"] = EWCTask(options=self.hp.ewc)
        return tasks
    
    def get_loss(self, forward_pass, rewards=None, loss_name=''):
        return super().get_loss(forward_pass, rewards=rewards, loss_name=loss_name)
        


@register_method
@dataclass
class EwcMethod(BaselineMethod, target_setting=IncrementalSetting):
    """ Method that adds the EWC Auxiliary Task to the `BaselineModel`. """
    hparams: EwcModel.HParams = mutable_field(EwcModel.HParams)

    def __init__(
        self,
        hparams: BaselineModel.HParams = None,
        config: Config = None,
        trainer_options: TrainerConfig = None,
        **kwargs,
    ):
        super().__init__(
            hparams=hparams, config=config, trainer_options=trainer_options, **kwargs
        )

    def configure(self, setting: Setting):
        """ Called before the method is applied on a setting (before training). 

        You can use this to instantiate your model, for instance, since this is
        where you get access to the observation & action spaces.
        """
        super().configure(setting)
        # self.model.add_auxiliary_task(EWCTask(options=self.hparams.ewc))

    def on_task_switch(self, task_id: Optional[int]):
        super().on_task_switch(task_id)

    def create_model(self, setting: Setting) -> EwcModel:
        """Create the Model to use for the given Setting.
        
        In this case, we want to return an `EwcModel` (our customized version of the
        BaselineModel).

        Parameters
        ----------
        setting : Setting
            The experimental Setting this Method will be applied to.

        Returns
        -------
        EwcModel
            The Model that will be trained and used for evaluation. 
        """
        return EwcModel(setting=setting, hparams=self.hparams, config=self.config)



def demo():
    # Adding arguments for each group directly:
    parser = ArgumentParser(description=__doc__)

    EwcMethod.add_argparse_args(parser, dest="method")
    parser.add_arguments(Config, "config")

    args = parser.parse_args()

    method = EwcMethod.from_argparse_args(args, dest="method")
    config: Config = args.config
    
    task_schedule = {
        0: {"gravity": 10, "length": 0.2},
        1000: {"gravity": 100, "length": 1.2},
        # 2000:   {"gravity": 10, "length": 0.2},
    }
    setting = TaskIncrementalRLSetting(
        dataset="cartpole",
        train_task_schedule=task_schedule,
        test_task_schedule=task_schedule,
        observe_state_directly=True,
        # max_steps=1000,
    )

    # setting = ClassIncrementalSetting(dataset="mnist", nb_tasks=5)
    # setting = TaskIncrementalSetting(dataset="mnist", nb_tasks=5)
    results = setting.apply(method)
    print(results.summary())


if __name__ == "__main__":
    demo()
