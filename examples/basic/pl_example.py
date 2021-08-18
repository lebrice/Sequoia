"""A simple example for creating a Method using PyTorch-Lightning.

Run this as:

```console
$> python examples/basic/pl_examples.py
```
"""
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import torch
from gym import spaces
from pytorch_lightning import LightningModule, Trainer
from torch import Tensor, nn
from torch.optim import Adam

from sequoia.common.config import Config
from sequoia.common.spaces import Image
from sequoia.methods import Method
from sequoia.settings.assumptions.task_type import ClassificationActions
from sequoia.settings.sl.continual import (
    Actions,
    ContinualSLSetting,
    Observations,
    ObservationSpace,
    Rewards,
)


class Model(LightningModule):
    """ Example Pytorch Lightning model used for continual image classification.

    Used by the `ExampleMethod` below.
    """

    @dataclass
    class HParams:
        """ Hyper-parameters of our model.

        NOTE: dataclasses are totally optional. This is just much nicer than dicts or
        ugly namespaces.
        """

        # Learning rate.
        learning_rate: float = 1e-3
        # Maximum number of training epochs per task.
        max_epochs_per_task: int = 1

    def __init__(
        self,
        input_space: ObservationSpace,
        output_space: spaces.Discrete,
        hparams: HParams = None,
    ):
        super().__init__()
        hparams = hparams or self.HParams()
        # NOTE: `input_space` is a subclass of `gym.spaces.Dict`. It contains (at least)
        # the `x` key, but can also contain other things, for example the task labels.
        # Doing things this way makes sure that this Model can also be applied to any
        # more specific Setting in the future (any setting with more information given)!
        image_space: Image = input_space.x
        # NOTE: `Image` is just a subclass of `gym.spaces.Box` with a few extra properties

        self.input_dims = image_space.shape
        # NOTE: Can't set the `hparams` attribute in PL, so use hp instead:
        self.hp = hparams
        self.save_hyperparameters({"hparams": asdict(hparams)})
        in_channels: int = image_space.channels
        num_classes: int = output_space.n

        # Imitates the SimpleConvNet from  sequoia.common.models.simple_convnet
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.AdaptiveAvgPool2d(output_size=(8, 8)),  # [16, 8, 8]
            # [32, 6, 6]
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # [32, 4, 4]
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.Flatten(),
        )
        # Quick tip: In this case we have a fixed hidden size (thanks to the Adaptive
        # pooling layer above), but you could also use the cool new `nn.LazyLinear` when
        # you don't know the hidden size in advance!
        self.fc = nn.Sequential(
            nn.Flatten(),
            # nn.LazyLinear(out_features=120),
            nn.Linear(512, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )
        self.loss = nn.CrossEntropyLoss()
        self.trainer: Trainer

    def forward(self, observations: ContinualSLSetting.Observations) -> Tensor:
        """Returns the logits for the given observation.

        Parameters
        ----------
        observations : ContinualSLSetting.Observations
            dataclass with (at least) the following attributes:
            - "x" (Tensor): the samples (images)
            - "task_labels" (Optional[Tensor]): Task labels, when applicable.

        Returns
        -------
        Tensor
            Classification logits for each class.
        """
        x: Tensor = observations.x
        # Task labels for each sample. We don't use them in this example.
        t: Optional[Tensor] = observations.task_labels
        h_x = self.features(x)
        logits = self.fc(h_x)
        return logits

    def training_step(
        self, batch: Tuple[Observations, Optional[Rewards]], batch_idx: int
    ) -> Tensor:
        return self.shared_step(batch=batch, batch_idx=batch_idx, stage="train")

    def validation_step(
        self, batch: Tuple[Observations, Optional[Rewards]], batch_idx: int
    ) -> Tensor:
        return self.shared_step(batch=batch, batch_idx=batch_idx, stage="val")

    def test_step(
        self, batch: Tuple[Observations, Optional[Rewards]], batch_idx: int
    ) -> Tensor:
        return self.shared_step(batch=batch, batch_idx=batch_idx, stage="test")

    def shared_step(
        self, batch: Tuple[Observations, Optional[Rewards]], batch_idx: int, stage: str,
    ) -> Tensor:
        observations, rewards = batch

        logits = self(observations)
        y_pred = logits.argmax(-1)
        actions = ClassificationActions(y_pred=y_pred, logits=logits)

        if rewards is None:
            environment: ContinualSLSetting.Environment
            # The rewards (image labels) might not be given at the same time as the
            # observations (images), for example during testing, or if we're being
            # evaluated based on our online performance during training!
            #
            # When that is the case, we need to send the "action" (predictions) to the
            # environment using `send()` to get the rewards.
            actions = y_pred
            # Get the current environment / dataloader from the Trainer.
            environment = self.trainer.request_dataloader(self, stage)
            rewards = environment.send(actions)
        y: Tensor = rewards.y

        accuracy = (y_pred == y).int().sum() / len(y)
        self.log(f"{stage}/accuracy", accuracy, prog_bar=True)

        loss = self.loss(logits, y)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hp.learning_rate)


class ExampleMethod(Method, target_setting=ContinualSLSetting):
    """ Example method for solving Continual SL Settings with PyTorch-Lightning

    This ExampleMethod declares that it can be applied to any `Setting` that inherits
    from this `ContinualSLSetting`.

    NOTE: Settings in Sequoia are a subclass of `LightningDataModule`, which create
    the training/validation/testing `Environment`s that methods will interact with.
    Each setting defines an `apply` method, which serves as a "main loop", and describes
    when and on what data to train the Method, and how it will be evaluated, according
    to the usual methodology for that setting in the litterature.

    Importantly, settings do NOT describe **how** the method is to be trained, that is
    entirely up to the Method! 
    """

    def __init__(self, hparams: Model.HParams = None):
        super().__init__()
        self.hparams = hparams or Model.HParams()
        self.current_task: Optional[int] = None
        # NOTE: These get assigned in `configure` below:
        self.model: Model
        self.trainer: Trainer

    def configure(self, setting: ContinualSLSetting):
        """ Called by the Setting so the method can configure itself before training.

        This could be used to, for example, create a model, since the observation space
        (which describes the types and shapes of the data) and the `nb_tasks` can be
        read from the Setting.

        Parameters
        ----------
        setting : ContinualSLSetting
            The research setting that this `Method` will be applied to.
        """
        if not setting.known_task_boundaries_at_train_time:
            # If we're being applied on a Setting where we don't have access to task
            # boundaries, then there is only one training environment that transitions
            # between all tasks and then closes itself.
            # We therefore limit the number of epochs per task to 1 in that case.
            self.hparams.max_epochs_per_task = 1
        self.model = Model(
            input_space=setting.observation_space,
            output_space=setting.action_space,
            hparams=self.hparams,
        )

    def fit(
        self,
        train_env: ContinualSLSetting.Environment,
        valid_env: ContinualSLSetting.Environment,
    ):
        """ Called by the Setting to allow the method to train.

        The passed environments inherit from `DataLoader` as well as from `gym.Env`.
        They produce `Observations` (which have an `x` Tensor field, for instance), and 
        return `Rewards` when they receive `Actions`.
        This interface is the same between RL and SL, making it easy to create methods
        that can adapt to both domains.

        Parameters
        ----------
        train_env : ContinualSLSetting.Environment
            The Training environment. In the case of a `ContinualSLSetting`, this
            environment will smoothly transition between the different tasks.
            NOTE: Regardless of what exact type of `Setting` this method is being
            applied to, this environment will always be a subclass of
            `ContinualSLSetting.Environment`, and the `Observations`, `Actions`,
            `Rewards` produced by this environment will also always follow this
            hierarchy.
            This is important to note, since it makes it possible to create a Method
            that also works in other settings which add extra information in the
            observations (e.g. task labels)!

        valid_env : ContinualSLSetting.Environment
            The Validation environment.
        """
        # NOTE: Currently have to 'reset' the Trainer for each call to `fit`.
        self.trainer = Trainer(
            gpus=torch.cuda.device_count(), max_epochs=self.hparams.max_epochs_per_task,
        )
        self.trainer.fit(
            self.model, train_dataloader=train_env, val_dataloaders=valid_env
        )

    def test(self, test_env: ContinualSLSetting.Environment):
        """ Called to let the Method handle the test loop by itself.

        The `test_env` will only give back rewards (y) once an action (y_pred) is sent
        to it via its `send` method.

        This test environment keeps track of some metrics of interest for its `Setting`
        (accuracy in this case) and reports them back to the `Setting` once the test
        environment has been exhausted.

        NOTE: The test environment will close itself when done, signifying the end
        of the test period. At that point, `test_env.is_closed()` will return `True`.
        """
        # BUG: There is currently a bug with the test loop with Trainer: on_task_switch
        # doesn't get called properly.
        raise NotImplementedError
        # Use ckpt_path=None to use the current weights, rather than the "best" ones.
        self.trainer.test(self.model, ckpt_path=None, test_dataloaders=test_env)

    def get_actions(
        self, observations: Observations, action_space: spaces.MultiDiscrete
    ):
        """ Called by the Setting to query for individual predictions.

        You currently have to implement this, but if `test` is implemented, it will be
        used instead. Sorry if this isn't super clear.
        """
        self.model.eval()
        with torch.no_grad():
            logits = self.model(observations.to(self.model.device))
            y_pred = logits.argmax(-1)
        return Actions(y_pred=y_pred)

    def on_task_switch(self, task_id: Optional[int]) -> None:
        """ Can be called by the Setting when a task boundary is reached.

        This will be called if `setting.known_task_boundaries_at_[train/test]_time` is
        True, depending on if this is called during training or during testing.

        If `setting.task_labels_at_[train/test]_time` is True, then `task_id` will be
        the identifyer (index) of the next task. If the value is False, then `task_id`
        will be None.
        """
        if task_id != self.current_task:
            phase = "training" if self.training else "testing"
            print(f"Switching tasks during {phase}: {self.current_task} -> {task_id}")
            self.current_task = task_id


def main():
    """ Runs the example: applies the method on a Continual Supervised Learning Setting.
    """
    # You could use any of the settings in SL, since this example methods targets the
    # most general Continual SL Setting in Sequoia: `ContinualSLSetting`:
    # from sequoia.settings.sl import ClassIncrementalSetting

    # Create the Setting:
    # NOTE: Since our model above uses an adaptive pooling layer, it should work on any
    # dataset!
    setting = ContinualSLSetting(dataset="mnist", monitor_training_performance=True)

    # Create the Method:
    method = ExampleMethod()

    # Create a config for the experiment (just so we can set a few options for this
    # example)
    config = Config(debug=True, log_dir="results/pl_example")

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
