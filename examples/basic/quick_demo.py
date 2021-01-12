""" Demo: Creates a simple new method and applies it to a single CL setting.
"""
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Type

from collections import defaultdict
from pathlib import Path

import gym
import pandas as pd
import tqdm
import torch
from numpy import inf
from gym import spaces
from torch import Tensor, nn
from simple_parsing import ArgumentParser


# This "hack" is required so we can run `python examples/quick_demo.py`
sys.path.extend([".", ".."])
from sequoia import Method, Setting
from sequoia.common import Config
from sequoia.settings import ClassIncrementalSetting
from sequoia.settings.passive.cl.objects import (Actions, Observations,
                                         PassiveEnvironment, Results, Rewards)


class MyModel(nn.Module):
    """ Simple classification model without any CL-related mechanism.

    To keep things simple, this demo model is designed for supervised
    (classification) settings where observations have shape [3, 28, 28] (ie the
    MNIST variants: Mnist, FashionMnist, RotatedMnist, EMnist, etc.)
    """
    def __init__(self,
                 observation_space: gym.Space,
                 action_space: gym.Space,
                 reward_space: gym.Space):
        super().__init__()
        image_shape = observation_space[0].shape
        assert image_shape == (3, 28, 28)
        assert isinstance(action_space, spaces.Discrete)
        assert action_space == reward_space
        n_classes = action_space.n
        image_channels = image_shape[0]

        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, n_classes),
        )
        self.loss = nn.CrossEntropyLoss()

    def forward(self, observations: Observations) -> Tensor:
        # NOTE: here we don't make use of the task labels.
        x = observations.x
        task_labels = observations.task_labels
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits

    def training_step(self, batch: Tuple[Observations, Rewards], *args, **kwargs):
        return self.shared_step(batch, *args, **kwargs)

    def validation_step(self, batch: Tuple[Observations, Rewards], *args, **kwargs):
        return self.shared_step(batch, *args, **kwargs)

    def shared_step(self, batch: Tuple[Observations, Rewards], *args, **kwargs):
        # Since we're training on a Passive environment, we get both
        # observations and rewards.
        observations: Observations = batch[0]
        rewards: Rewards = batch[1]
        image_labels = rewards.y

        # Get the predictions:
        logits = self(observations)
        y_pred = logits.argmax(-1)

        loss = self.loss(logits, image_labels)

        accuracy = (y_pred == image_labels).sum().float() / len(image_labels)
        metrics_dict = {"accuracy": accuracy}
        return loss, metrics_dict


class DemoMethod(Method, target_setting=ClassIncrementalSetting):
    """ Minimal example of a Method targetting the Class-Incremental CL setting.
    
    For a quick intro to dataclasses, see examples/dataclasses_example.py    
    """

    @dataclass
    class HParams:
        """ Hyper-parameters of the demo model. """
        # Learning rate of the optimizer.
        learning_rate: float = 0.001
        
        @classmethod
        def from_args(cls) -> "HParams":
            """ Get the hparams of the method from the command-line. """
            parser = ArgumentParser(description=cls.__doc__)
            parser.add_arguments(cls, dest="hparams")
            args, _ = parser.parse_known_args()
            return args.hparams

    def __init__(self, hparams: HParams = None):
        self.hparams: DemoMethod.HParams = hparams or self.HParams.from_args()
        self.max_epochs: int = 1
        self.early_stop_patience: int = 2

        # We will create those when `configure` will be called, before training.
        self.model: MyModel
        self.optimizer: torch.optim.Optimizer

    def configure(self, setting: ClassIncrementalSetting):
        """ Called before the method is applied on a setting (before training). 

        You can use this to instantiate your model, for instance, since this is
        where you get access to the observation & action spaces.
        """
        self.model = MyModel(
            observation_space=setting.observation_space,
            action_space=setting.action_space,
            reward_space=setting.reward_space,
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)

    def fit(self, train_env: PassiveEnvironment, valid_env: PassiveEnvironment):
        """ Example train loop.
        You can do whatever you want with train_env and valid_env here.
        
        NOTE: In the Settings where task boundaries are known (in this case all
        the supervised CL settings), this will be called once per task.
        """
        # configure() will have been called by the setting before we get here.
        best_val_loss = inf
        best_epoch = 0
        for epoch in range(self.max_epochs):
            self.model.train()
            print(f"Starting epoch {epoch}")
            # Training loop:
            with tqdm.tqdm(train_env) as train_pbar:
                postfix = {}
                train_pbar.set_description(f"Training Epoch {epoch}")
                for i, batch in enumerate(train_pbar):
                    loss, metrics_dict = self.model.training_step(batch)
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
                epoch_val_loss = 0.

                for i, batch in enumerate(val_pbar):
                    batch_val_loss, metrics_dict = self.model.validation_step(batch)
                    epoch_val_loss += batch_val_loss
                    postfix.update(metrics_dict, val_loss=epoch_val_loss)
                    val_pbar.set_postfix(postfix)
            torch.set_grad_enabled(True)

            if epoch_val_loss < best_val_loss:
                best_val_loss = valid_env
                best_epoch = i
            if i - best_epoch > self.early_stop_patience:
                print(f"Early stopping at epoch {i}.")

    def get_actions(self, observations: Observations, action_space: gym.Space) -> Actions:
        """ Get a batch of predictions (aka actions) for these observations. """ 
        with torch.no_grad():
            logits = self.model(observations)
        # Get the predicted classes
        y_pred = logits.argmax(dim=-1)
        return self.target_setting.Actions(y_pred)

    @classmethod
    def add_argparse_args(cls, parser: ArgumentParser, dest: str = ""):
        """Adds command-line arguments for this Method to an argument parser."""
        parser.add_arguments(cls.HParams, "hparams")
        
    @classmethod
    def from_argparse_args(cls, args, dest=None):
        """Creates an instance of this Method from the parsed arguments."""
        hparams: cls.HParams = args.hparams
        return cls(hparams=hparams)


def demo_simple():
    """ Simple demo: Creating and applying a Method onto a Setting. """
    from sequoia.settings import TaskIncrementalSetting
    
    ## 1. Creating the setting:
    setting = TaskIncrementalSetting(dataset="fashionmnist", batch_size=32)
    ## 2. Creating the Method
    method = DemoMethod()
    # (Optional): You can also create a Config, which holds other fields like
    # `log_dir`, `debug`, `device`, etc. which aren't specific to either the
    # Setting or the Method.
    config = Config(debug=True, render=True)
    ## 3. Applying the method to the setting: (optionally passing a Config to
    # use for that run)
    results = setting.apply(method, config=config)
    print(results.summary())
    print(f"objective: {results.objective}")


def demo_command_line():
    """ Run this quick demo from the command-line. """
    from sequoia.settings import ClassIncrementalSetting, TaskIncrementalSetting
    parser = ArgumentParser(description=__doc__)
    # Add command-line arguments for the Method and the Setting.
    DemoMethod.add_argparse_args(parser)
    # Add command-line arguments for the Setting and the Config (an object with
    # options like log_dir, debug, etc, which are not part of the Setting or the
    # Method) using simple-parsing.
    parser.add_arguments(TaskIncrementalSetting, "setting")
    parser.add_arguments(Config, "config")
    args = parser.parse_args()
    
    # Create the Method from the parsed arguments
    method: DemoMethod = DemoMethod.from_argparse_args(args)
    # Extract the Setting and Config from the args.
    setting: TaskIncrementalSetting = args.setting
    config: Config = args.config
    
    # Run the demo, applying that DemoMethod on the given setting.
    results: Results = setting.apply(method, config=config)
    print(results.summary())
    print(f"objective: {results.objective}")
    results.save_to_dir(config.log_dir)


if __name__ == "__main__":
    # Example: Evaluate a Method on a single CL setting:
    
    ###
    ### First option: Run the demo, creating the Setting and Method directly.
    ###
    demo_simple()
    
    ##
    ## Second part of the demo: Same as before, but customize the options for
    ## the Setting and the Method from the command-line.
    ##
    
    # demo_command_line()
    
    ##
    ## As a little bonus: Evaluate on *ALL* the applicable settings, and
    ## aggregate the results in a nice little LaTeX-formatted table.
    ##
    
    # from examples.demo_utils import demo_all_settings
    # all_results = demo_all_settings(DemoMethod)
