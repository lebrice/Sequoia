""" Demo script. """
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, Optional, Tuple, Type, List

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
from methods import MethodABC as MethodBase
from settings import (ClassIncrementalSetting, PassiveEnvironment, Results,
                      Setting)
from utils.utils import prod


Observations = ClassIncrementalSetting.Observations
Actions = ClassIncrementalSetting.Actions
Rewards = ClassIncrementalSetting.Rewards


class MyNewMethod(MethodBase, target_setting=ClassIncrementalSetting):
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
        from pytorch_lightning.loggers import WandbLogger

        setting_name: str = setting.get_name()
        method_name: str = self.name
        dataset: str = setting.dataset
        
        logger = WandbLogger(
            name=f"{self.name}-{setting_name}-{dataset}",
            save_dir="results",
            project="demo",
            group=f"{setting_name}-{dataset}",
        )
        logger.experiment.summary["method"] = method_name
        logger.experiment.summary["setting"] = setting_name
        logger.experiment.summary["dataset"] = dataset
        self.trainer = pl.Trainer(
            gpus=1,
            max_epochs=1,
            logger=logger,
            # fast_dev_run=True,
        )
        self.model = self.create_model(setting)

    def create_model(self, setting: ClassIncrementalSetting):
        return MyModel(
            observation_space=setting.observation_space,
            action_space=setting.action_space,
            reward_space=setting.reward_space,
            **self.hparams.to_dict()
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
        assert self.trainer and self.model
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
        self.observation_space = observation_space
        self.action_space = action_space
        self.reward_space = reward_space
        self.learning_rate = learning_rate
        print(f"Observation space: {self.observation_space}")
        print(f"Action space: {self.action_space}")
        print(f"Reward space: {self.action_space}")

        # Pytorch-Lightning, saves arguments of the constructor above.
        self.save_hyperparameters()
        
        # In Class-Incremental setting, the observation space is a Tuple
        # containing the image and the task label.
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
        assert in_channels in {1, 3}, f"should only be 1 or 3 input channels {image_space}"
        
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
        return self.net(x)

    def get_predictions(self, observations):
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

        return {
            "loss": loss,
            "log": {"accuracy": accuracy},
            "progress_bar": {"accuracy": accuracy}
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def demo(method_type = MyNewMethod) -> Dict[Type[Setting], Dict[str, Results]]:
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
            # # Apply the method on the setting.
            # results: Results = setting.apply(method, config=config)
            # all_results[SettingClass][dataset] = results
            
            # Use this (and comment out below) to debug just the tables below:
            class FakeResult:
                objective: float = 1.23
            all_results[SettingClass][dataset] = FakeResult()
             
            # print(f"Results for Method {method.get_name()} on setting {SettingClass}, dataset {dataset}:")
            # print(results.summary())
            # wandb.log(results.to_log_dict())
            # wandb.log(results.make_plots())
            # wandb.summary["method"] = method.get_name()
            # wandb.summary["setting"] = setting.get_name()
            # wandb.summary["dataset"] = dataset
            # wandb.summary[results.objective_name] = results.objective

    print(f"----- All Results for method {method_type} -------")
    # Create a LaTeX table with all the results for all the settings.
    import pandas as pd
    
    all_settings: List[Type[Setting]] = list(all_results.keys())
    all_setting_names: List[str] = [s.get_name() for s in all_settings]

    all_datasets: List[str] = []
    for setting, dataset_to_results in all_results.items():
        all_datasets.extend(dataset_to_results.keys())                
    all_datasets = list(set(all_datasets))
    
    ## Create a multi-index for the dataframe.
    # tuples = []
    # for setting, dataset_to_results in all_results.items():
    #     setting_name = setting.get_name()
    #     tuples.extend((setting_name, dataset) for dataset in dataset_to_results.keys())
    # tuples = sorted(list(set(tuples)))
    # multi_index = pd.MultiIndex.from_tuples(tuples, names=["setting", "dataset"])
    # single_index = pd.Index(["Objective"])
    # df = pd.DataFrame(index=multi_index, columns=single_index)

    df = pd.DataFrame(index=all_setting_names, columns=all_datasets)

    for setting_type, dataset_to_results in all_results.items():
        setting_name = setting_type.get_name()
        for dataset, result in dataset_to_results.items():
            # df["Objective"][setting_name, dataset] = result.objective
            df[dataset][setting_name] = result.objective

    caption = f"Results for method {method_type.__name__} on all its applicable settings."
    print(df)

    results_csv_path = Path(f"examples/results/results_{method_type.get_name()}.csv")
    latex_table_path = Path(f"examples/results/table_{method_type.get_name()}.tex")
    results_csv_path.parent.mkdir(exist_ok=True, parents=True)

    with open(results_csv_path, "w") as f:
        df.to_csv(f)
    print(f"Saved dataframe with results to path {results_csv_path}")
    
    with open(latex_table_path, "w") as f:
        print(df.to_latex(
            caption=caption,
            na_rep="N/A",
            multicolumn=True,
            # sparsify=True,
            # header=False,
            # columns=False,
        ), file=f)
    print(f"Saved LaTeX table with results to path {latex_table_path}")
    return all_results


if __name__ == "__main__":
    demo()
