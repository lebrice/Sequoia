""" Demo script. """
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Tuple, Type

import gym
import numpy as np
import pytorch_lightning as pl
import torch
import tqdm
import wandb
from gym import spaces
from pytorch_lightning.core.decorators import auto_move_data
from simple_parsing import ArgumentParser, Serializable, choice
from torch import Tensor, nn

from common.config import Config
from methods import MethodABC as MethodBase
from settings import (ClassIncrementalSetting, PassiveEnvironment, Results,
                      Setting)
from methods.models import BaseHParams
from utils import dict_union

import pandas as pd
from .demo_utils import make_result_dataframe, save_results_table 

Observations = ClassIncrementalSetting.Observations
Actions = ClassIncrementalSetting.Actions
Rewards = ClassIncrementalSetting.Rewards

class SimpleConvNet(nn.Module):
    def __init__(self, in_channels=3, n_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, n_classes),
        )
    def forward(self, x: Tensor) -> Tensor:
        return self.fc(self.features(x))


class DemoMethod(MethodBase, target_setting=ClassIncrementalSetting):
    @dataclass
    class HParams(BaseHParams):
        """ HyperParameters of this new method.
        To see all the builtin hyperparameters, check out the BaseHParams class.
        """
        # Register a simple new encoder, in addition to the resnets and other
        # torchvision models already available.
        available_encoders = dict_union(BaseHParams.available_encoders, {"simple_convnet": SimpleConvNet})
        # Use an encoder architecture from the torchvision.models package.
        encoder: str = choice(available_encoders.keys(), default="simple_convnet")

    def __init__(self, hparams: HParams, config: Config):
        self.hparams: DemoMethod.HParams = hparams
        self.config = config
        self.max_epochs: int = 1
        self.early_stop_patience: int = 2

        self.model: MyModel
        self.optimizer: torch.optim.Optimizer

    def configure(self, setting: ClassIncrementalSetting):
        """ Configure this method before being applied on a given setting. """
        self.model = MyModel(
            observation_space=setting.observation_space,
            action_space=setting.action_space,
            reward_space=setting.reward_space,
            hparams=self.hparams,
        )
        self.optimizer = self.model.configure_optimizers()

    def fit(self, train_env: PassiveEnvironment, valid_env: PassiveEnvironment):
        # configure() will have been called by the setting before we get here.
        best_val_loss = np.inf
        best_epoch = 0
        for epoch in range(self.max_epochs):
            self.model.train()
            print(f"Starting epoch {epoch}")
            # Training loop:
            with tqdm.tqdm(train_env) as train_pbar:
                train_pbar.set_description(f"Training Epoch {epoch}")
                for i, batch in enumerate(train_pbar):
                    loss, metrics_dict = self.model.training_step(batch)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    train_pbar.set_postfix(**metrics_dict)

            # Validation loop:
            self.model.eval()
            torch.set_grad_enabled(False)
            with tqdm.tqdm(valid_env) as val_pbar:
                val_pbar.set_description(f"Validation Epoch {epoch}")
                epoch_val_loss = 0.

                for i, batch in enumerate(val_pbar):
                    batch_val_loss, metrics_dict = self.model.validation_step(batch)
                    epoch_val_loss += batch_val_loss
                    val_pbar.set_postfix(**metrics_dict, val_loss=epoch_val_loss)
            torch.set_grad_enabled(True)

            if epoch_val_loss < best_val_loss:
                best_val_loss = valid_env
                best_epoch = i
            if i - best_epoch > self.early_stop_patience:
                print(f"Early stopping at epoch {i}.")

    def get_actions(self, observations: Observations, action_space: gym.Space) -> Actions:
        with torch.no_grad():
            y_pred = self.model.get_predictions(observations)          
        return self.target_setting.Actions(y_pred)


class MyModel(nn.Module):
    def __init__(self,
                 observation_space: gym.Space,
                 action_space: gym.Space,
                 reward_space: gym.Space,
                 hparams: DemoMethod.HParams):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.reward_space = reward_space
        self.hparams = hparams

        # This demo is meant for classification problems atm.
        assert isinstance(self.action_space, spaces.Discrete)
        self.output_shape = (self.action_space.n,)
        self.loss = nn.CrossEntropyLoss()
        
        image_space: spaces.Box = self.observation_space[0]
        in_channels = image_space.shape[0]
        n_outputs = np.prod(self.output_shape, dtype=int)
        # Create an encoder and get the resulting hidden size.
        self.encoder, hidden_size = self.hparams.make_encoder(self.hparams.encoder)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_size, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, n_outputs),
        )

    @auto_move_data
    def forward(self, observations: Observations) -> Tensor:
        x, task_labels = observations
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits

    def get_predictions(self, observations: Observations) -> Tensor:
        """ Get a batch of predictions (aka actions) for these observations. """ 
        logits = self(observations)
        actions = logits.argmax(dim=-1)
        return actions

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
        loss = self.loss(logits, image_labels)

        predicted_labels = logits.argmax(-1)
        accuracy = (predicted_labels == image_labels).sum().float() / len(image_labels)
        metrics_dict = {"accuracy": accuracy}
        return loss, metrics_dict

    def configure_optimizers(self):
        return self.hparams.make_optimizer(self.parameters())



def demo(method_type=DemoMethod):
    parser = ArgumentParser(description=__doc__)
    parser.add_arguments(Config, dest="config", default=Config())
    parser.add_arguments(method_type.HParams, dest="hparams")

    args = parser.parse_args()

    config: Config = args.config
    hparams: method_type.HParams = args.hparams
    method = method_type(hparams=hparams, config=config)

    all_results: Dict[Type[Setting], Dict[str, Results]] = {}
    
    for SettingClass in method.get_applicable_settings():
        all_results[SettingClass] = {}
        
        # for dataset in SettingClass.available_datasets:
        for dataset in ["mnist", "fashion_mnist"]:
            # Instantiate the Setting, using the default options for each
            # setting, for instance the number of tasks, etc.
            setting = SettingClass(dataset=dataset)
            # Apply the method on the setting.
            results: Results = setting.apply(method, config=config)
            all_results[SettingClass][dataset] = results
            
            # Use this (and comment out below) to debug just the tables below:
            # class FakeResult:
            #     objective: float = 1.23
            # all_results[SettingClass][dataset] = FakeResult()
             
            print(f"Results for Method {method.get_name()} on setting {SettingClass}, dataset {dataset}:")
            print(results.summary())

    result_df: pd.DataFrame = make_result_dataframe(all_results)

    csv_path = Path(f"examples/results/results_{method_type.get_name()}.csv")
    latex_table_path = Path(f"examples/results/table_{method_type.get_name()}.tex")

    caption = f"Results for method {method_type.__name__} on all its applicable settings."
    
    csv_path.parent.mkdir(exist_ok=True, parents=True)
    with open(csv_path, "w") as f:
        result_df.to_csv(f)
    print(f"Saved dataframe with results to path {csv_path}")
    result_df.to_latex(
        buf=latex_table_path,
        caption=caption,
        na_rep="N/A",
        multicolumn=True,
    )
    print(f"Saved LaTeX table with results to path {latex_table_path}")
   
if __name__ == "__main__":
    demo(DemoMethod)
