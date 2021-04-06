""" Example Method for the SL track: Uses a simple classifier, without any CL mechanism.

As you'd expect, this Method exhibits complete forgetting of all previous tasks.
You can use this model and method as a jumping off point for your own submission.
"""
from dataclasses import dataclass
from typing import ClassVar, Dict, List, Optional, Tuple, Type

import gym
import torch
import tqdm
from gym import spaces
from numpy import inf
from simple_parsing import ArgumentParser
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer
from torchvision.models import ResNet, resnet18

from sequoia.common.hparams import HyperParameters, log_uniform
from sequoia.common.spaces import Image
from sequoia.methods import Method
from sequoia.settings import ClassIncrementalSetting
from sequoia.settings.passive import PassiveEnvironment
from sequoia.settings.passive.cl.objects import (Actions, Environment,
                                                 Observations, Rewards)

@dataclass
class HParams(HyperParameters):
    """ Hyper-parameters of the demo model. """

    # Learning rate of the optimizer.
    learning_rate: float = log_uniform(1e-6, 1e-2, default=0.001)
    # L2 regularization coefficient.
    weight_decay: float = log_uniform(1e-9, 1e-3, default=1e-6)

    # Maximum number of training epochs per task.
    max_epochs_per_task: int = 10
    # Number of epochs with increasing validation loss after which we stop training.
    early_stop_patience: int = 2


class Classifier(nn.Module):
    """ Simple classification model without any CL-related mechanism.

    This example model uses a resnet18 as the encoder, and a single output layer.
    """
    
    HParams: ClassVar[Type[HParams]] = HParams
    
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        reward_space: gym.Space,
        hparams: HParams = None,
    ):
        super().__init__()
        self.hparams = hparams or self.HParams()

        image_space: Image = observation_space.x
        # image_shape = image_space.shape

        # This example is intended for classification / discrete action spaces.
        assert isinstance(action_space, spaces.Discrete)
        assert action_space == reward_space
        self.n_classes = action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder, self.representations_size = self.create_encoder(image_space)
        self.output = self.create_output_head()
        self.loss = nn.CrossEntropyLoss()

    def create_output_head(self) -> nn.Module:
        return nn.Linear(self.representations_size, self.n_classes).to(self.device)

    def configure_optimizers(self) -> Optimizer:
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

    def create_encoder(self, image_space: Image) -> Tuple[nn.Module, int]:
        """Create an encoder for the given image space.

        Returns the encoder, as well as the size of the representations it will produce.

        Parameters
        ----------
        image_space : Image
            A subclass of `gym.spaces.Box` for images. Represents the space the images
            will come from during training and testing. Its attributes of interest
            include `c`, `w`, `h`, `shape` and `dype`.

        Returns
        -------
        Tuple[nn.Module, int]
            The encoder to be used, (a nn.Module), as well as the size of the
            representations it will produce.

        Raises
        ------
        NotImplementedError
            If no encoder is available for the given image dimensions.
        """
        if image_space.width == image_space.height == 28:
            # Setup for mnist variants.
            # (not part of the competition, but used for debugging below).
            encoder = nn.Sequential(
                nn.Conv2d(image_space.channels, 6, 5),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(6, 16, 5),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
            )
            features = 256
        elif image_space.width == image_space.height == 32:
            # Synbols dataset: use a resnet18 by default.
            resnet: ResNet = resnet18(pretrained=False)
            features = resnet.fc.in_features
            # Disable/Remove the last layer.
            resnet.fc = nn.Sequential()
            encoder = resnet
        else:
            raise NotImplementedError(
                f"TODO: Add an encoder for the given image space {image_space}"
            )
        return encoder.to(self.device), features

    def forward(self, observations: Observations) -> Tensor:
        # NOTE: here we don't make use of the task labels.
        observations = observations.to(self.device)
        x = observations.x
        task_labels = observations.task_labels
        features = self.encoder(x)
        logits = self.output(features)
        return logits

    def shared_step(
        self, batch: Tuple[Observations, Optional[Rewards]], environment: Environment
    ) -> Tuple[Tensor, Dict]:
        """Shared step used for both training and validation.

        Parameters
        ----------
        batch : Tuple[Observations, Optional[Rewards]]
            Batch containing Observations, and optional Rewards. When the Rewards are
            None, it means that we'll need to provide the Environment with actions
            before we can get the Rewards (e.g. image labels) back.

            This happens for example when being applied in a Setting which cares about
            sample efficiency or training performance, for example.

        environment : Environment
            The environment we're currently interacting with. Used to provide the
            rewards when they aren't already part of the batch (as mentioned above).

        Returns
        -------
        Tuple[Tensor, Dict]
            The Loss tensor, and a dict of metrics to be logged.
        """
        # Since we're training on a Passive environment, we will get both observations
        # and rewards, unless we're being evaluated based on our training performance,
        # in which case we will need to send actions to the environments before we can
        # get the corresponding rewards (image labels).
        observations: Observations = batch[0]
        rewards: Optional[Rewards] = batch[1]
        # Get the predictions:
        logits = self(observations)
        y_pred = logits.argmax(-1)

        if rewards is None:
            # If the rewards in the batch is None, it means we're expected to give
            # actions before we can get rewards back from the environment.
            rewards = environment.send(Actions(y_pred))

        assert rewards is not None
        image_labels = rewards.y.to(self.device)

        loss = self.loss(logits, image_labels)

        accuracy = (y_pred == image_labels).sum().float() / len(image_labels)
        metrics_dict = {"accuracy": f"{accuracy.cpu().item():3.2%}"}
        return loss, metrics_dict


class ExampleMethod(Method, target_setting=ClassIncrementalSetting):
    """ Minimal example of a Method usable only in the SL track of the competition.

    This method uses the ExampleModel, which is quite simple.
    """
    ModelType: ClassVar[Type[Classifier]] = Classifier
    
    def __init__(self, hparams: HParams = None):
        self.hparams: HParams = hparams or HParams()

        # We will create those when `configure` will be called, before training.
        self.model: Classifier
        self.optimizer: torch.optim.Optimizer

    def configure(self, setting: ClassIncrementalSetting):
        """ Called before the method is applied on a setting (before training).

        You can use this to instantiate your model, for instance, since this is
        where you get access to the observation & action spaces.
        """
        self.model = self.ModelType(
            observation_space=setting.observation_space,
            action_space=setting.action_space,
            reward_space=setting.reward_space,
        )
        self.optimizer = self.model.configure_optimizers()
    
    def fit(self, train_env: PassiveEnvironment, valid_env: PassiveEnvironment):
        """ Example train loop.
        You can do whatever you want with train_env and valid_env here.

        NOTE: In the Settings where task boundaries are known (in this case all
        the supervised CL settings), this will be called once per task.
        """
        # configure() will have been called by the setting before we get here.
        best_val_loss = inf
        best_epoch = 0
        for epoch in range(self.hparams.max_epochs_per_task):
            self.model.train()
            print(f"Starting epoch {epoch}")
            # Training loop:
            with tqdm.tqdm(train_env) as train_pbar:
                postfix = {}
                train_pbar.set_description(f"Training Epoch {epoch}")
                for i, batch in enumerate(train_pbar):
                    loss, metrics_dict = self.model.shared_step(
                        batch, environment=train_env
                    )
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    postfix.update(metrics_dict)
                    train_pbar.set_postfix(postfix)

            # Validation loop:
            self.model.eval()
            torch.set_grad_enabled(False)
            with tqdm.tqdm(valid_env) as val_pbar:
                postfix = {}
                val_pbar.set_description(f"Validation Epoch {epoch}")
                epoch_val_loss = 0.0

                for i, batch in enumerate(val_pbar):
                    batch_val_loss, metrics_dict = self.model.shared_step(
                        batch, environment=valid_env
                    )
                    epoch_val_loss += batch_val_loss
                    postfix.update(metrics_dict, val_loss=epoch_val_loss)
                    val_pbar.set_postfix(postfix)
            torch.set_grad_enabled(True)

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_epoch = epoch
            if epoch - best_epoch > self.hparams.early_stop_patience:
                print(f"Early stopping at epoch {i}.")
                # NOTE: You should probably reload the model weights as they were at the
                # best epoch.
                break

    def get_actions(
        self, observations: Observations, action_space: gym.Space
    ) -> Actions:
        """ Get a batch of predictions (aka actions) for these observations. """
        with torch.no_grad():
            logits = self.model(observations)
        # Get the predicted classes
        y_pred = logits.argmax(dim=-1)
        return self.target_setting.Actions(y_pred)

    @classmethod
    def add_argparse_args(cls, parser: ArgumentParser, dest: str = ""):
        """Adds command-line arguments for this Method to an argument parser."""
        parser.add_arguments(cls.ModelType.HParams, "hparams")

    @classmethod
    def from_argparse_args(cls, args, dest: str = ""):
        """Creates an instance of this Method from the parsed arguments."""
        hparams: Classifier.HParams = args.hparams
        return cls(hparams=hparams)


if __name__ == "__main__":
    from sequoia.common import Config
    from sequoia.settings import ClassIncrementalSetting
    # Create the Method:
    # - Manually:
    # method = ExampleMethod()
    # - From the command-line:
    from simple_parsing import ArgumentParser

    parser = ArgumentParser()
    ExampleMethod.add_argparse_args(parser)
    args = parser.parse_args()
    method = ExampleMethod.from_argparse_args(args)

    # Create the Setting:

    # - "Easy": Domain-Incremental MNIST Setting, useful for quick debugging, but
    #           beware that the action space is different than in class-incremental!
    #           (which is the type of Setting used in the SL track!)
    # from sequoia.settings.passive.cl.domain_incremental import DomainIncrementalSetting
    # setting = DomainIncrementalSetting(
    #     dataset="mnist", nb_tasks=5, monitor_training_performance=True
    # )

    # - "Medium": Class-Incremental MNIST Setting, useful for quick debugging:
    # setting = ClassIncrementalSetting(
    #     dataset="mnist",
    #     nb_tasks=5,
    #     monitor_training_performance=True,
    #     known_task_boundaries_at_test_time=False,
    #     batch_size=32,
    #     num_workers=4,
    # )

    # - "HARD": Class-Incremental Synbols, more challenging.
    # NOTE: This Setting is very similar to the one used for the SL track of the
    # competition.
    setting = ClassIncrementalSetting(
        dataset="synbols",
        nb_tasks=12,
        known_task_boundaries_at_test_time=False,
        monitor_training_performance=True,
        batch_size=32,
        num_workers=4,
    )
    # NOTE: can also use pass a `Config` object to `setting.apply`. This object has some
    # configuration options like device, data_dir, etc.
    results = setting.apply(method, config=Config(data_dir="data"))
    print(results.summary())
