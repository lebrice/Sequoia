""" Demo: Creates a simple new method and applies it to various CL settings. """
from dataclasses import dataclass
from typing import Dict, Tuple, Type

import gym
import torch
from gym import spaces
from torch import Tensor, nn

from settings import MethodABC as Method
from settings import Setting
from settings.passive.cl import ClassIncrementalSetting
from settings.passive.cl.objects import (Actions, Observations,
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
        x, task_labels = observations
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
    
    def __init__(self, hparams: HParams):
        self.hparams: DemoMethod.HParams = hparams
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
        # configure() will have been called by the setting before we get here.
        import tqdm
        from numpy import inf
        best_val_loss = inf
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
        """ Get a batch of predictions (aka actions) for these observations. """ 
        with torch.no_grad():
            logits = self.model(observations)
        # Get the predicted classes
        y_pred = logits.argmax(dim=-1)
        return self.target_setting.Actions(y_pred)


def demo(method_type = DemoMethod):
    method = create_method(method_type)
    all_results = evaluate_on_all_settings(method)
    return all_results


def create_method(DemoMethod = DemoMethod) -> DemoMethod:
    from simple_parsing import ArgumentParser
    
    # Get the hparams of the method from the command-line.
    parser = ArgumentParser(description=__doc__)
    parser.add_arguments(DemoMethod.HParams, dest="hparams")
    args = parser.parse_args()
    hparams: DemoMethod.HParams = args.hparams

    # Create the method and return it.
    method = DemoMethod(hparams=hparams)
    return method


def evaluate_on_all_settings(method: DemoMethod, below: Type[Setting]=None):
    """ Applies the method to all its applicable settings and shows the results.
    """
    import pandas as pd
    from collections import defaultdict
    from pathlib import Path

    # Iterate over all the applicable evaluation settings, using the default
    # options for each setting, and store the results inside this dictionary.
    all_results: Dict[Type[Setting], Dict[str, Results]] = defaultdict(dict)
    
    setting: ClassIncrementalSetting
    for setting in method.all_evaluation_settings():
        if below is not None:
            if not isinstance(setting, below):
                continue

        setting_type = type(setting)         
        dataset = setting.dataset

        # Limiting this demo to just mnist/fashion_mnist datasets.
        if setting.dataset not in ["mnist", "fashion_mnist"]:
            # print(f"Skipping {setting_type} / {setting.dataset} for now.")
            continue

        # Apply the method on the setting.
        results: Results = setting.apply(method)
        print(f"Results on setting {setting_type}, dataset {dataset}:")
        print(results.summary())
        # Save the results.
        all_results[setting_type][dataset] = results

    # Aggregate all the results in a pandas DataFrame.

    from .demo_utils import make_result_dataframe
    result_df: pd.DataFrame = make_result_dataframe(all_results)

    csv_path = Path(f"examples/results/results_{method.get_name()}.csv")
    csv_path.parent.mkdir(exist_ok=True, parents=True)
    result_df.to_csv(csv_path)
    print(f"Saved dataframe with results to path {csv_path}")

    # BONUS: Display the results in a LaTeX-formatted table!

    latex_table_path = Path(f"examples/results/table_{method.get_name()}.tex")
    caption = f"Results for method {type(method).__name__} settings."
    result_df.to_latex(
        buf=latex_table_path,
        caption=caption,
        na_rep="N/A",
        multicolumn=True,
    )
    print(f"Saved LaTeX table with results to path {latex_table_path}")
    return all_results


if __name__ == "__main__":
    demo(DemoMethod)
