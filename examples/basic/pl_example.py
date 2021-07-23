from pytorch_lightning import LightningModule, Trainer
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from sequoia.settings.sl import SLSetting
from sequoia.common.spaces import TypedDictSpace, TensorBox, TensorDiscrete
import torch
from dataclasses import dataclass, asdict
from torch import Tensor
from typing import TypedDict, Tuple, Optional, Union, List
from gym import spaces
from sequoia.settings.assumptions.task_type import ClassificationActions
from sequoia.methods import Method
from torch import nn, Tensor
from torch.optim import Optimizer, Adam
from sequoia.common.spaces import Image
from sequoia.settings.sl.continual import (
    Observations,
    Actions,
    Rewards,
    ContinualSLSetting,
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

        lr: float = 1e-3

    def __init__(
        self,
        input_space: TypedDict,
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
        # NOTE: In this case we have a fixed hidden size (thanks to the Adaptive pooling
        # above), but you can also use `nn.LazyLinear` when you don't know the hidden
        # size in advance!
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
        return Adam(self.parameters(), lr=self.hp.lr)


class ExampleMethod(Method, target_setting=SLSetting):
    def __init__(self, hparams: Model.HParams = None):
        super().__init__()
        self.hparams = hparams
        self.current_task: Optional[int] = None

        self.model: Model
        self.trainer: Trainer

    def configure(self, setting: ContinualSLSetting):
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
        self.model.train_env = train_env
        self.model.valid_env = valid_env
        # NOTE: Currently have to 'reset' the Trainer for each call to `fit`.
        self.trainer = Trainer(gpus=torch.cuda.device_count(), max_epochs=1, accelerator="ddp")
        self.trainer.fit(
            self.model, train_dataloader=train_env, val_dataloaders=valid_env
        )

    def test(self, test_env: ContinualSLSetting.Environment):
        """ Called to let the Method handle the test loop by itself.
        
        NOTE: The `test_env` will not give back rewards (y) until an action (y_pred) is
        sent to it via the `send` method.
        """
        # for obs, rewards in test_env:
        #     assert rewards is None
        #     batch_size = obs.batch_size
        #     # NOTE: Last batch is often shorter, just watch out for that!
        #     action = test_env.action_space.sample()
        #     action = action[:batch_size]
        #     rewards = test_env.send(action)
        self.model.test_env = test_env
        self.trainer.test(self.model, ckpt_path=None, test_dataloaders=test_env)

    def get_actions(
        self, observations: Observations, action_space: spaces.MultiDiscrete
    ):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(observations)
            y_pred = logits.argmax(-1)
        return Actions(y_pred=y_pred)

    def on_task_switch(self, task_id: Optional[int]) -> None:
        """ Can be called by the Setitng when a task boundary is reached.
        
        This will be called if `setting.known_task_boundaries_at_[train/test]_time` is
        True, depending on if this is called during training or during testing.
        """
        if task_id != self.current_task:
            phase = "training" if self.training else "testing"
            print(f"Switching tasks during {phase}: {self.current_task} -> {task_id}")
            self.current_task = task_id


if __name__ == "__main__":
    from sequoia.settings.sl import ClassIncrementalSetting, ContinualSLSetting
    from sequoia.common.config import Config
    setting = ContinualSLSetting(dataset="mnist", monitor_training_performance=True)
    method = ExampleMethod()

    # Create a config for the experiment
    config = Config(debug=True, log_dir="results/pl_example")

    results = setting.apply(method, config=config)
    print(results.summary())

    for figure_name, figure in results.make_plots().items():
        print("Figure:", figure_name)
        figure.show()
        # figure.waitforbuttonpress(10)
