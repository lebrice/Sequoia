""" Demo script. """
from dataclasses import dataclass
from typing import Optional, Tuple

import gym
import numpy as np
import pytorch_lightning as pl
import torch
from gym import spaces
from simple_parsing import ArgumentParser, Serializable
from torch import Tensor, nn

from common.config import Config
from common.metrics import (ClassificationMetrics, Metrics, RegressionMetrics,
                            get_metrics)
from methods import AbstractMethod, Method
from settings import (ClassIncrementalSetting, PassiveEnvironment, Results,
                      Setting)
from utils.utils import prod

Observations = ClassIncrementalSetting.Observations
Actions = ClassIncrementalSetting.Actions
Rewards = ClassIncrementalSetting.Rewards


class MyNewMethod(AbstractMethod, target_setting=ClassIncrementalSetting):
    """ Example of writing a new method. """

    @dataclass
    class HParams(Serializable):
        """ Example of HyperParameters of this method. """
        learning_rate: float = 0.01

    def __init__(self, hparams: HParams, config: Config):
        self.hparams = hparams
        self.config = config
        
        self.trainer: pl.Trainer
        self.model: MyModel

    def configure(self, setting: Setting):
        """ Configure this method before being applied on a setting. """
        self.trainer = pl.Trainer(
            gpus=1,
            max_epochs=1,
            fast_dev_run=True,
        )
        self.model = MyModel(
            setting=setting,
            observation_space=setting.observation_space,
            action_space=setting.action_space,
            reward_space=setting.reward_space,
            **self.hparams.to_dict()
        )

    def fit(self,
            train_dataloader: PassiveEnvironment = None,
            valid_dataloader: PassiveEnvironment = None,
            datamodule: pl.LightningDataModule = None):        
        # configure() will have been called by the setting before we get here.
        assert self.trainer and self.model
        env = train_dataloader
        self.trainer.fit(
            model=self.model,
            train_dataloader=train_dataloader,
            val_dataloaders=valid_dataloader,
            datamodule=datamodule,
        )

    def get_actions(self, observations: Observations, action_space: gym.Space) -> Actions:
        with torch.no_grad():
            y_pred = self.model.get_predictions(observations)
        return self.target_setting.Actions(y_pred)
        return super().get_actions(observations, action_space)


class MyModel(pl.LightningModule):
    def __init__(self,
                 setting: Setting,
                 observation_space: gym.Space,
                 action_space: gym.Space,
                 reward_space: gym.Space,
                 learning_rate: float = 3e-4):
        super().__init__()
        self.setting: ClassIncrementalSetting = setting
        self.observation_space = observation_space
        self.action_space = action_space
        self.reward_space = reward_space
        self.learning_rate = learning_rate
        print(f"observation_space: {observation_space}")
        print(f"action_space: {action_space}")
        
        assert isinstance(observation_space, spaces.Tuple)
        image_space: spaces.Box = observation_space[0]
        input_features = np.prod(image_space.shape, dtype=int)
        
        if isinstance(action_space, spaces.Discrete):
            # Classification problem
            self.output_shape = (action_space.n,)
            self.loss = nn.CrossEntropyLoss()
        elif isinstance(action_space, spaces.Box):
            # Regression problem
            self.output_shape = action_space.shape
            self.loss = nn.MSELoss()
            
        output_features = np.prod(self.output_shape, dtype=int)
        
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_features, output_features),
        )

    def forward(self, observations) -> Tensor:
        x, task_labels = observations
        return self.net(x)

    def get_predictions(self, observations):
        """ Get a batch of predictions (aka actions) for these observations. """ 
        logits = self(observations)
        actions = logits.argmax(-1)
        return actions

    def training_step(self, batch: Tuple[Observations, Rewards], batch_idx: int, **kwargs):
        # Since we're training, we get both observations and rewards.
        observations: ClassIncrementalSetting.Observations = batch[0]
        rewards: ClassIncrementalSetting.Rewards = batch[1]

        image_labels = rewards.y
        # Get the predictions:
        logits = self(observations)

        loss = self.loss(logits, image_labels)
        predicted_labels = logits.argmax(-1)
        accuracy = (predicted_labels == image_labels).sum().float() / len(image_labels)
        metrics: Metrics = get_metrics(y_pred=logits, y=image_labels)

        # TrainResult auto-detaches the loss after the optimization steps are complete
        return {
            "loss": loss,
            "log": metrics.to_log_dict(),
            "progress_bar": metrics.to_pbar_message(),
        }
        # result = pl.TrainResult(minimize=loss)
        # result.log("accuracy", accuracy, prog_bar=True)
        # return result

    def validation_step(self, batch: Tuple[Observations, Rewards], batch_idx: int, **kwargs):
        # Since we're training, we get both observations and rewards.
        observations: ClassIncrementalSetting.Observations = batch[0]
        rewards: ClassIncrementalSetting.Rewards = batch[1]

        image_labels = rewards.y
        # Get the predictions:
        logits = self(observations)

        loss = self.loss(logits, image_labels)
        predicted_labels = logits.argmax(-1)
        accuracy = (predicted_labels == image_labels).sum().float() / len(image_labels)
        metrics: Metrics = get_metrics(y_pred=logits, y=image_labels)

        # TrainResult auto-detaches the loss after the optimization steps are complete
        return {
            "loss": loss,
            "log": metrics.to_log_dict(),
            "progress_bar": metrics.to_pbar_message(),
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_arguments(Config, dest="config", default=Config())
    parser.add_arguments(MyNewMethod.HParams, dest="hparams")
    
    # pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    # pl.Trainer.from_argparse_args(args)
    
    config: Config = args.config
    
    hparams: MyNewMethod.HParams = args.hparams
    method = MyNewMethod(hparams=hparams, config=config)

    for SettingClass in method.get_applicable_settings():
        print(f"Type of Setting: {SettingClass}")
        
        # for dataset in SettingClass.available_datasets:
        # Instantiate the Setting.
        dataset = "mnist"
        setting = SettingClass(dataset=dataset)

        # Apply the method on the setting.
        results: Results = setting.apply(method, config=config)
        
        print(f"Results for setting {SettingClass}, dataset {dataset}:")
        print(results.summary())
