from typing import Optional

import torch
from dataclasses import dataclass
from simple_parsing import mutable_field

from sequoia.common import Config
from sequoia.methods import BaseModel
from sequoia.methods.trainer import Trainer, TrainerConfig
from sequoia.settings.sl import ContinualSLSetting, TaskIncrementalSLSetting
from examples.basic.pl_example import ExampleMethod, Model
from sequoia.methods.packnet_method import PackNet


class ExamplePackNetMethod(ExampleMethod, target_setting=TaskIncrementalSLSetting):
    def __init__(
        self, hparams: Model.HParams = None, packnet_hparams: PackNet.HParams = None
    ):
        super().__init__(hparams=hparams)
        self.packnet_hparams = packnet_hparams or PackNet.HParams()
        # TODO: Modify `hparams.max_epochs_per_task` to at least be enough so that
        # PackNet will work.
        min_epochs = (
            self.packnet_hparams.train_epochs + self.packnet_hparams.fine_tune_epochs
        )
        if self.hparams.max_epochs_per_task < min_epochs:
            self.hparams.max_epochs_per_task = min_epochs
        self.p_net: PackNet

    def configure(self, setting: TaskIncrementalSLSetting):
        super().configure(setting)
        # TODO: Why does PackNet need access to the number of tasks again?
        self.p_net = PackNet(
            n_tasks=setting.nb_tasks,
            hparams=self.packnet_hparams,
        )
        # TODO: This could be set as default values in the PackNet constructor.
        self.p_net.current_task = -1
        self.p_net.config_instructions()

    def fit(
        self,
        train_env: TaskIncrementalSLSetting.Environment,
        valid_env: TaskIncrementalSLSetting.Environment,
    ):
        # NOTE: PackNet is not compatible with EarlyStopping, thus we set max_epochs==min_epochs
        self.trainer = Trainer(
            gpus=torch.cuda.device_count(),
            min_epochs=self.p_net.total_epochs(),
            max_epochs=self.p_net.total_epochs(),
            callbacks=[self.p_net]
        )

        self.trainer.fit(
            self.model, train_dataloader=train_env, val_dataloaders=valid_env
        )

    def on_task_switch(self, task_id: Optional[int]):
        """Called when switching between tasks.

        Args:
            task_id (int, optional): the id of the new task. When None, we are
            basically being informed that there is a task boundary, but without
            knowing what task we're switching to.
        """
        super().on_task_switch(task_id=task_id)
        if task_id is not None and len(self.p_net.masks) > task_id:
            self.p_net.load_final_state(model=self.model)
            self.p_net.apply_eval_mask(task_idx=task_id, model=self.model)
        self.p_net.current_task = task_id


def main():
    """Runs the example: applies the method on a Continual Supervised Learning Setting."""
    # You could use any of the settings in SL, since this example methods targets the
    # most general Continual SL Setting in Sequoia: `ContinualSLSetting`:
    # from sequoia.settings.sl import ClassIncrementalSetting

    # Create the Setting:
    # NOTE: Since our model above uses an adaptive pooling layer, it should work on any
    # dataset!
    setting = TaskIncrementalSLSetting(
        dataset="mnist", nb_tasks=5, monitor_training_performance=True
    )

    # Create the Method:
    method = ExamplePackNetMethod()

    # Create a config for the experiment (just so we can set a few options for this
    # example)
    config = Config(debug=False, log_dir="results/pl_example_packnet")

    # Launch the experiment: trains and tests the method according to the chosen
    # setting and returns a Results object.
    results = setting.apply(method, config=config)

    # Print the results, and show some plots!
    print(results.summary())
    for figure_name, figure in results.make_plots().items():
        print("Figure:", figure_name)
        figure.show()
        # figure.waitforbuttonpress(10)


if __name__ == "__main__":
    main()
