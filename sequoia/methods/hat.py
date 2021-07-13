""" Hard Attention to the Task

```
@inproceedings{serra2018overcoming,
    title={Overcoming Catastrophic Forgetting with Hard Attention to the Task},
    author={Serra, Joan and Suris, Didac and Miron, Marius and Karatzoglou, Alexandros},
    booktitle={International Conference on Machine Learning},
    pages={4548--4557},
    year={2018}
}
```
"""

from dataclasses import dataclass
from typing import Dict, NamedTuple, Optional, Tuple, Mapping, Union, Any

import gym
import numpy as np
import torch
import tqdm
from numpy import inf
from simple_parsing import ArgumentParser
from torch import Tensor
from wandb.wandb_run import Run

from sequoia.methods import register_method
from sequoia.common import Config
from sequoia.common.hparams import HyperParameters, log_uniform, uniform, categorical
from sequoia.common.spaces import Image
from sequoia.settings import Method, Environment, Setting
from sequoia.settings.sl import TaskIncrementalSLSetting
from sequoia.settings.sl.incremental.objects import (
    Actions,
    Observations,
    Rewards,
)
from sequoia.settings.sl.environment import PassiveEnvironment


class Masks(NamedTuple):
    """ Named tuple for the masked tensors created in the HATNet. """

    gc1: Tensor
    gc2: Tensor
    gc3: Tensor
    gfc1: Tensor
    gfc2: Tensor


class HatNet(torch.nn.Module):
    """
    @inproceedings{serra2018overcoming,
      title={Overcoming Catastrophic Forgetting with Hard Attention to the Task},
      author={Serra, Joan and Suris, Didac and Miron, Marius and Karatzoglou, Alexandros},
      booktitle={International Conference on Machine Learning},
      pages={4548--4557},
      year={2018}
    }

    The model is where the model weights are initialized.
    Just like a classic PyTorch, here the different layers and components of the model
    are defined.
    """

    def __init__(
        self, image_space: Image, n_classes_per_task: Dict[int, int], s_hat: int = 50
    ):
        super().__init__()

        ncha = image_space.channels
        size = image_space.width
        self.n_classes_per_task = n_classes_per_task
        self.s_hat = s_hat

        self.c1 = torch.nn.Conv2d(ncha, 64, kernel_size=size // 8)
        s = compute_conv_output_size(size, size // 8)
        s //= 2
        self.c2 = torch.nn.Conv2d(64, 128, kernel_size=size // 10)
        s = compute_conv_output_size(s, size // 10)
        s //= 2
        self.c3 = torch.nn.Conv2d(128, 256, kernel_size=2)
        s = compute_conv_output_size(s, 2)
        s //= 2
        self.smid = s
        self.maxpool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()

        self.drop1 = torch.nn.Dropout(0.2)
        self.drop2 = torch.nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(256 * self.smid * self.smid, 2048)
        self.fc2 = torch.nn.Linear(2048, 2048)
        self.output_layers = torch.nn.ModuleList()

        n_tasks = len(self.n_classes_per_task)
        # TODO: (@lebrice) Here I'm 'fixing' this, by making it so each output head has
        # as many outputs as there are classes in total. It's not super efficient, but
        # it should work.
        total_classes = sum(self.n_classes_per_task.values())
        for task_index, n_classes_in_task in self.n_classes_per_task.items():
            self.output_layers.append(torch.nn.Linear(2048, total_classes))

        self.gate = torch.nn.Sigmoid()
        # All embedding stuff should start with 'e'
        self.ec1 = torch.nn.Embedding(n_tasks, 64)
        self.ec2 = torch.nn.Embedding(n_tasks, 128)
        self.ec3 = torch.nn.Embedding(n_tasks, 256)
        self.efc1 = torch.nn.Embedding(n_tasks, 2048)
        self.efc2 = torch.nn.Embedding(n_tasks, 2048)

        self.flatten = torch.nn.Flatten()

        self.loss = torch.nn.CrossEntropyLoss()
        self.current_task: Optional[int] = 0

    def forward(
        self, observations: TaskIncrementalSLSetting.Observations
    ) -> Tuple[Tensor, Masks]:
        observations.as_list_of_tuples()
        x = observations.x
        t = observations.task_labels
        # BUG: This won't work if task_labels is None (which is the case at
        # test-time in the ClassIncrementalSetting)
        masks = self.mask(t, s_hat=self.s_hat)
        gc1, gc2, gc3, gfc1, gfc2 = masks
        # Gated
        h = self.maxpool(self.drop1(self.relu(self.c1(x))))
        h = h * gc1.unsqueeze(2).unsqueeze(3)
        h = self.maxpool(self.drop1(self.relu(self.c2(h))))
        h = h * gc2.unsqueeze(2).unsqueeze(3)
        h = self.maxpool(self.drop2(self.relu(self.c3(h))))
        h = h * gc3.unsqueeze(2).unsqueeze(3)
        h = self.flatten(h)
        h = self.drop2(self.relu(self.fc1(h)))
        h = h * gfc1.expand_as(h)
        h = self.drop2(self.relu(self.fc2(h)))
        h = h * gfc2.expand_as(h)

        # Each batch can have elements of more than one Task (in test)
        # In Task Incremental Learning, each task have it own classification head.
        y: Optional[Tensor] = None
        task_masks = {}
        for task_id in set(t.tolist()):
            task_mask = t == task_id
            task_masks[task_id] = task_mask

            y_pred_t = self.output_layers[task_id](h.clone())
            if y is None:
                y = y_pred_t
            else:
                y[task_mask] = y_pred_t[task_mask]
        assert y is not None
        return y, masks

    def mask(self, t: Tensor, s_hat: float) -> Masks:
        gc1 = self.gate(s_hat * self.ec1(t))
        gc2 = self.gate(s_hat * self.ec2(t))
        gc3 = self.gate(s_hat * self.ec3(t))
        gfc1 = self.gate(s_hat * self.efc1(t))
        gfc2 = self.gate(s_hat * self.efc2(t))
        return Masks(gc1, gc2, gc3, gfc1, gfc2)

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
            rewards when they aren't already part of the batch, for example when our
            performance is being monitored during training.

        Returns
        -------
        Tuple[Tensor, Dict]
            The Loss tensor, and a dict of metrics to be logged.
        """
        # Since we're training on a Passive environment, we will get both observations
        # and rewards, unless we're being evaluated based on our training performance,
        # in which case we will need to send actions to the environments before we can
        # get the corresponding rewards (image labels) back.
        observations: Observations = batch[0]
        rewards: Optional[Rewards] = batch[1]

        # Get the predictions:
        logits, _ = self(observations)
        y_pred = logits.argmax(-1)

        if rewards is None:
            # If the rewards in the batch were None, it means we're expected to give
            # actions before we can get rewards back from the environment.
            # This happens when the Setting is monitoring our training performance.
            rewards = environment.send(Actions(y_pred))

        assert rewards is not None
        image_labels = rewards.y

        loss = self.loss(logits, image_labels)

        accuracy = (y_pred == image_labels).sum().float() / len(image_labels)
        metrics_dict = {"accuracy": accuracy}
        return loss, metrics_dict


def compute_conv_output_size(
    Lin: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1
) -> int:
    return int(
        np.floor(
            (Lin + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1
        )
    )


@register_method
class HatMethod(Method, target_setting=TaskIncrementalSLSetting):
    """Hard Attention to the Task

    ```
    @inproceedings{serra2018overcoming,
        title={Overcoming Catastrophic Forgetting with Hard Attention to the Task},
        author={Serra, Joan and Suris, Didac and Miron, Marius and Karatzoglou, Alexandros},
        booktitle={International Conference on Machine Learning},
        pages={4548--4557},
        year={2018}
    }
    ```
    """

    @dataclass
    class HParams(HyperParameters):
        """ Hyper-parameters of the Settings. """

        # Learning rate of the optimizer.
        learning_rate: float = log_uniform(1e-6, 1e-2, default=0.001)
        # Batch size
        batch_size: int = categorical(16, 32, 64, 128, default=128)
        # weight/importance of the task embedding to the gate function
        s_hat: float = uniform(1.0, 100.0, default=50.0)
        # Maximum number of training epochs per task
        max_epochs_per_task: int = uniform(1, 20, default=10, discrete=True)

    def __init__(self, hparams: HParams = None):
        self.hparams: HatMethod.HParams = hparams or self.HParams()
        self.early_stopping_patience = 2
        # We will create those when `configure` will be called, before training.
        self.model: HatNet
        self.optimizer: torch.optim.Optimizer

    def configure(self, setting: TaskIncrementalSLSetting):
        """ Called before the method is applied on a setting (before training).

        You can use this to instantiate your model, for instance, since this is
        where you get access to the observation & action spaces.
        """
        setting.batch_size = self.hparams.batch_size
        assert (
            setting.increment == setting.test_increment
        ), "Assuming same number of classes per task for training and testing."
        n_classes_per_task = {
            i: setting.num_classes_in_task(i, train=True)
            for i in range(setting.nb_tasks)
        }
        image_space: Image = setting.observation_space["x"]
        self.model = HatNet(
            image_space=image_space,
            n_classes_per_task=n_classes_per_task,
            s_hat=self.hparams.s_hat,
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.hparams.learning_rate,
        )

    def fit(self, train_env: PassiveEnvironment, valid_env: PassiveEnvironment):
        """
        Train loop

        Different Settings can return elements from tasks in an other  way,
        be it class incremental, task incremental, etc.

        Batch can have information about en environment, rewards, input, task labels, etc.
        And we call the forward training function of our method, independent of the settings
        """

        # configure() will have been called by the setting before we get here,

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
                        batch, environment=train_env,
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
                        batch, environment=valid_env,
                    )
                    epoch_val_loss += batch_val_loss
                    postfix.update(metrics_dict, val_loss=epoch_val_loss)
                    val_pbar.set_postfix(postfix)
            torch.set_grad_enabled(True)

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_epoch = i
            elif epoch - best_epoch > self.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    def get_actions(
        self, observations: Observations, action_space: gym.Space
    ) -> Actions:
        """ Get a batch of predictions (aka actions) for these observations. """
        with torch.no_grad():
            logits, _ = self.model(observations)
        # Get the predicted classes
        y_pred = logits.argmax(dim=-1)
        return self.target_setting.Actions(y_pred)

    def on_task_switch(self, task_id: Optional[int]):
        # This method gets called if task boundaries are known in the current
        # setting. Furthermore, if task labels are available, task_id will be
        # the index of the new task. If not, task_id will be None.
        # TODO: Does this method actually work when task_id is None?
        self.model.current_task = task_id

    @classmethod
    def add_argparse_args(cls, parser: ArgumentParser, dest: str = "") -> None:
        parser.add_arguments(cls.HParams, dest="hparams")
        # You can also add arguments as usual:
        # parser.add_argument("--foo", default=123)

    @classmethod
    def from_argparse_args(cls, args, dest: str = "") -> "HatMethod":
        hparams: HatMethod.HParams = args.hparams
        # foo: int = args.foo
        method = cls(hparams=hparams)
        return method

    def get_search_space(self, setting: Setting) -> Mapping[str, Union[str, Dict]]:
        """Returns the search space to use for HPO in the given Setting.

        Parameters
        ----------
        setting : Setting
            The Setting on which the run of HPO will take place.

        Returns
        -------
        Mapping[str, Union[str, Dict]]
            An orion-formatted search space dictionary, mapping from hyper-parameter
            names (str) to their priors (str), or to nested dicts of the same form.
        """
        return self.hparams.get_orion_space()

    def adapt_to_new_hparams(self, new_hparams: Dict[str, Any]) -> None:
        """Adapts the Method when it receives new Hyper-Parameters to try for a new run.

        It is required that this method be implemented if you want to perform HPO sweeps
        with Orion.

        Parameters
        ----------
        new_hparams : Dict[str, Any]
            The new hyper-parameters being recommended by the HPO algorithm. These will
            have the same structure as the search space.
        """
        # Here we overwrite the corresponding attributes with the new suggested values
        # leaving other fields unchanged.
        # NOTE: These new hyper-paramers will be used in the next run in the sweep,
        # since each call to `configure` will create a new Model.
        self.hparams = self.hparams.replace(**new_hparams)

    def setup_wandb(self, run: Run) -> None:
        """ Called by the Setting when using Weights & Biases, after `wandb.init`.

        This method is here to provide Methods with the opportunity to log some of their
        configuration options or hyper-parameters to wandb.

        NOTE: The Setting has already set the `"setting"` entry in the `wandb.config` by
        this point.

        Parameters
        ----------
        run : wandb.Run
            Current wandb Run.
        """
        run.config["hparams"] = self.hparams.to_dict()


if __name__ == "__main__":
    # Example: Evaluate a Method on a single CL setting:
    parser = ArgumentParser(description=__doc__, add_dest_to_option_strings=False)

    """
    We must define 3 main components:
     1.- Setting: It is the continual learning scenario that we are working, SL or RL, TI or CI
                  Each settings has it own parameters that can be customized.
     2.- Model: Is the parameters and layers of the model, just like in PyTorch.
                We can use a predefined model or create your own
     3.- Method: It is how we are going to use what the settings give us to train our model.
                 Same as before, we can define our own or use pre-defined Methods.
    """
    # Add arguments for the Method, the Setting, and the Config.
    # (Config contains options like the log_dir, the data_dir, etc.)
    HatMethod.add_argparse_args(parser, dest="method")
    parser.add_arguments(TaskIncrementalSLSetting, dest="setting")
    parser.add_arguments(Config, "config")

    args = parser.parse_args()

    # Create the Method from the args, and extract the Setting, and the Config:
    method: HatMethod = HatMethod.from_argparse_args(args, dest="method")
    setting: TaskIncrementalSLSetting = args.setting
    config: Config = args.config

    # Apply the method to the setting, optionally passing in a Config,
    # producing Results.
    results = setting.apply(method, config=config)
    print(results.summary())
    print(f"objective: {results.objective}")
