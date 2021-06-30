"""Defines the EWC method, as a subclass of the BaseMethod.

Likewise, defines the `EwcModel`, which is a very simple subclass of the
`BaselineModel`, adding in the Ewc auxiliary task (`EWCTask`).

For a more detailed view of exactly how the EwcTask calculates its loss, see
the `sequoia.methods.aux_tasks.ewc.EwcTask`.
"""
import warnings
from dataclasses import dataclass
from typing import Optional

from gym.utils import colorize
from simple_parsing import ArgumentParser, mutable_field

from sequoia.common.config import Config
from sequoia.common.config.trainer_config import TrainerConfig
from sequoia.methods import register_method
from sequoia.methods.aux_tasks.ewc import EWCTask
from sequoia.methods.base_method import BaseMethod, BaselineModel
from sequoia.settings import Setting, TaskIncrementalRLSetting
from sequoia.settings.assumptions.incremental import IncrementalAssumption


class EwcModel(BaselineModel):
    """ Modified version of the BaselineModel, which adds the EWC auxiliary task. """

    @dataclass
    class HParams(BaselineModel.HParams):
        """ Hyper-parameters of the `EwcModel`. """

        # Hyper-parameters related to the EWC auxiliary task.
        ewc: EWCTask.Options = mutable_field(EWCTask.Options)

    def __init__(self, setting: Setting, hparams: "EwcModel.HParams", config: Config):
        super().__init__(setting=setting, hparams=hparams, config=config)
        self.hp: EwcModel.HParams
        self.add_auxiliary_task(EWCTask(options=self.hp.ewc))

    def get_loss(self, forward_pass, rewards=None, loss_name=""):
        return super().get_loss(forward_pass, rewards=rewards, loss_name=loss_name)


@register_method
@dataclass
class EwcMethod(BaseMethod, target_setting=IncrementalAssumption):
    """ Subclass of the BaseMethod, which adds the EWCTask to the `BaselineModel`.

    This Method is applicable to any CL setting (RL or SL) where there are clear task
    boundaries, regardless of if the task labels are given or not.
    """

    hparams: EwcModel.HParams = mutable_field(EwcModel.HParams)

    def __init__(
        self,
        hparams: EwcModel.HParams = None,
        config: Config = None,
        trainer_options: TrainerConfig = None,
        **kwargs,
    ):
        super().__init__(
            hparams=hparams, config=config, trainer_options=trainer_options, **kwargs
        )

    def configure(self, setting: IncrementalAssumption):
        """ Called before the method is applied on a setting (before training).

        You can use this to instantiate your model, for instance, since this is
        where you get access to the observation & action spaces.
        """
        super().configure(setting)

        if setting.phases == 1:
            warnings.warn(
                RuntimeWarning(
                    colorize(
                        "Disabling the EWC portion of this Method entirely, as there "
                        "is only one phase of training in this setting (i.e. `fit` is "
                        "only called once).",
                        "red",
                    )
                )
            )
            # We could also just disable the ewc task (after super().configure(setting))
            self.model.tasks["ewc"].disable()

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
    """ Runs the EwcMethod on a simple setting, just to check that it works fine.
    """

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
        # max_steps=1000,
    )

    # from sequoia.settings import TaskIncrementalSLSetting, ClassIncrementalSetting
    # setting = ClassIncrementalSetting(dataset="mnist", nb_tasks=5)
    # setting = TaskIncrementalSLSetting(dataset="mnist", nb_tasks=5)
    results = setting.apply(method, config=config)
    print(results.summary())


if __name__ == "__main__":
    demo()
