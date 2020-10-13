""" Demo script. """
from dataclasses import dataclass
from typing import ClassVar, Dict, Optional, Tuple, Type

import gym
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from gym import spaces
from pytorch_lightning.core.decorators import auto_move_data
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


class MyNewMethod(Method, target_setting=ClassIncrementalSetting):
    """ Example of writing a new method. """
    # Name for our new method.
    name: ClassVar[str] = "demo"

    @dataclass
    class HParams(Serializable):
        """ Example of HyperParameters of this method. """
        learning_rate: float = 0.01

    def __init__(self, hparams: HParams, config: Config):
        self.hparams = hparams
        self.config: Config = config
        
        self.trainer: pl.Trainer
        self.model: MyModel

    def configure(self, setting: Setting):
        """ Configure this method before being applied on a given setting. """
        setting_name: str = setting.get_name()
        method_name: str = self.get_name()
        dataset = setting.dataset
        from pytorch_lightning.loggers import WandbLogger
        logger = WandbLogger(
            name=f"{self.name}-{setting_name}-{dataset}",
            save_dir="results",
            project="demo",
            group=f"{setting_name}-{dataset}",
        )
        self.trainer = pl.Trainer(
            gpus=1,
            max_epochs=1,
            logger=logger,
            # fast_dev_run=True,
        )

    def fit(self,
            train_env: PassiveEnvironment = None,
            valid_env: PassiveEnvironment = None,
            datamodule: pl.LightningDataModule = None):
        """Train your method however you want, using the train and valid envs/dataloaders.

        Parameters
        ----------
        train_env : PassiveEnvironment, optional
            Hybrid of DataLoader & gym.Env which you can use to train your method.
            By default None.
        valid_env : PassiveEnvironment, optional
            Hybrid of DataLoader & gym.Env which you can use to train your method.
            By default None 
        """
        # configure() will have been called by the setting before we get here.
        
        self.model = MyModel(
            observation_space=train_env.observation_space,
            action_space=train_env.action_space,
            reward_space=train_env.reward_space,
            **self.hparams.to_dict()
        )
        self.trainer.fit(
            model=self.model,
            train_dataloader=train_env,
            val_dataloaders=train_env,
            datamodule=datamodule,
        )

    def get_actions(self, observations: Observations, action_space: gym.Space) -> Actions:
        with torch.no_grad():
            y_pred = self.model.get_predictions(observations)          
        return self.target_setting.Actions(y_pred)


class MyModel(pl.LightningModule):
    def __init__(self,
                 observation_space: gym.Space,
                 action_space: gym.Space,
                 reward_space: gym.Space,
                 learning_rate: float = 1e-4):
        super().__init__()
        # The spaces we're given have a batch dimension, so we just extract a
        # single slice.
        self.observation_space = observation_space[0]
        self.action_space = action_space[0]
        self.reward_space = reward_space[0]
        self.learning_rate = learning_rate
        print(f"Observation space: {self.observation_space}")
        print(f"Action space: {self.action_space}")
        print(f"Reward space: {self.action_space}")

        # In Class-Incremental setting, the observation space 
        assert isinstance(self.observation_space, spaces.Tuple)
        
        image_space: spaces.Box = self.observation_space[0]
        
        if isinstance(self.action_space, spaces.Discrete):
            # Classification problem
            self.output_shape = (self.action_space.n,)
            self.loss = nn.CrossEntropyLoss()
        elif isinstance(self.action_space, spaces.Box):
            # Regression problem
            self.output_shape = self.action_space.shape
            self.loss = nn.MSELoss()
        else:
            assert False, self.action_space

        in_channels = image_space.shape[0]
        assert in_channels in {1, 3}, "should only be 1 or 3 input channels."
        
        n_outputs = np.prod(self.output_shape, dtype=int)
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(256, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, n_outputs),
        )

    @auto_move_data
    def forward(self, observations) -> Tensor:
        x, task_labels = observations
        # assert False, self.net(x).shape
        return self.net(x)

    def get_predictions(self, observations):
        """ Get a batch of predictions (aka actions) for these observations. """ 
        logits = self(observations)
        actions = logits.argmax(dim=-1)
        # import matplotlib.pyplot as plt
        # plt.imshow(observations[0][0].permute(1, 2, 0))
        # plt.show()
        # assert False, (logits[0], actions[0])
        return actions

    def training_step(self, batch: Tuple[Observations, Rewards], batch_idx: int, **kwargs):
        return self.shared_step(batch, batch_idx)

    def validation_step(self, batch: Tuple[Observations, Rewards], batch_idx: int, **kwargs):
        return self.shared_step(batch, batch_idx)

    def shared_step(self, batch: Tuple[Observations, Rewards], batch_idx: int, **kwargs):
        # Since we're training, we get both observations and rewards.
        observations: Observations = batch[0]
        rewards: Rewards = batch[1]

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

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def demo():
    parser = ArgumentParser(description=__doc__)
    parser.add_arguments(Config, dest="config", default=Config())
    parser.add_arguments(MyNewMethod.HParams, dest="hparams")

    args = parser.parse_args()

    config: Config = args.config
    hparams: MyNewMethod.HParams = args.hparams

    method = MyNewMethod(hparams=hparams, config=config)

    all_results: Dict[Type[Setting], Dict[str, Results]] = {}
    
    for SettingClass in method.get_applicable_settings():
        all_results[SettingClass] = {}
        print(f"Type of Setting: {SettingClass}")
        
        # for dataset in SettingClass.available_datasets:
        for dataset in ["mnist", "fashion_mnist"]:
            # Instantiate the Setting, using the default options for each
            # setting, for instance the number of tasks, etc.
            setting = SettingClass(dataset=dataset)

            # Apply the method on the setting.
            results: Results = setting.apply(method, config=config)
            all_results[SettingClass][dataset] = results
            print(f"Results for setting {SettingClass}, dataset {dataset}:")
            print(results.summary())
            wandb.log(results.to_log_dict())
            wandb.log(results.make_plots())
            wandb.summary["method"] = method.get_name()
            wandb.summary["setting"] = setting.get_name()
            wandb.summary["dataset"] = dataset
            wandb.summary[results.objective_name] = results.objective

    print("----- All Results -------")

    for setting_type, dataset_to_results in all_results.items():
        print(f" ----- {setting_type} ------")
        objective_name = results.objective_name
        import pandas as pd
        
        datasets = []
        objectives = []
        for dataset, result in dataset_to_results.items():
            datasets.append(dataset)
            objectives.append(result.objective)
        
        results_df = pd.DataFrame({
            objective_name:     datasets,
            method.get_name(): objectives
        })
        print(results_df)
        print(results_df.to_latex(index=False))  


if __name__ == "__main__":
    demo()
