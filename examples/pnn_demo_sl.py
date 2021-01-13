import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from numpy import inf
from simple_parsing import ArgumentParser

from sequoia.common import Config
from sequoia.methods.stable_baselines3_methods import A2CModel
from sequoia.settings import (Method, TaskIncrementalRLSetting,
                              TaskIncrementalSetting)
from sequoia.settings.passive.cl import ClassIncrementalSetting
from sequoia.settings.passive.cl.objects import (Actions, Observations,
                                                 PassiveEnvironment, Rewards)

class PNNConvLayer(nn.Module):
    def __init__(self, col, depth, n_in, n_out):
        super(PNNConvLayer, self).__init__()
        self.col = col
        self.layer = nn.Conv2d(n_in, n_out, 3, stride=2, padding=1)

        self.u = nn.ModuleList()
        if depth > 0:
            self.u.extend([ nn.Conv2d(n_in, n_out, 3, stride=2, padding=1) for _ in range(col) ])

    def forward(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]

        cur_column_out = self.layer(inputs[-1])
        prev_columns_out = [mod(x) for mod, x in zip(self.u, inputs)]

        return F.elu(cur_column_out + sum(prev_columns_out))

class PNNGruLayer(nn.Module):
    def __init__(self, col, depth, n_in, n_out):
        super(PNNGruLayer, self).__init__()
        self.layer = nn.GRUCell(n_in, n_out) #nn.GRUCell(32 * 5 * 5, 256)

        #self.u = nn.ModuleList()
        #if depth > 0:
        #    self.u.extend([ nn.GRUCell(n_in, n_out) for _ in range(col) ])

    def forward(self, inputs, hx):
        #if not isinstance(inputs, list):
        #    inputs = [inputs]
        cur_column_out = self.layer(inputs, hx)
        # prev_columns_out = [mod(x, hx) for mod, x in zip(self.u, inputs)]

        return cur_column_out

class PNNLinearBlock(nn.Module):
    def __init__(self, col, depth, n_in, n_out):
        super(PNNLinearBlock, self).__init__()
        self.layer = nn.Linear(n_in, n_out)

        self.u = nn.ModuleList()
        if depth > 0:
            self.u.extend([nn.Linear(n_in, n_out) for _ in range(col)])

    def forward(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        cur_column_out = self.layer(inputs[-1])
        prev_columns_out = [mod(x) for mod, x in zip(self.u, inputs)]

        return F.relu(cur_column_out + sum(prev_columns_out))

class PNN(nn.Module):
    """
    @article{rusu2016progressive,
      title={Progressive neural networks},
      author={Rusu, Andrei A and Rabinowitz, Neil C and Desjardins, Guillaume and Soyer, Hubert and Kirkpatrick, James and Kavukcuoglu, Koray and Pascanu, Razvan and Hadsell, Raia},
      journal={arXiv preprint arXiv:1606.04671},
      year={2016}
    }
    """
    def __init__(self, n_layers):
        super(PNN, self).__init__()
        self.n_layers = n_layers
        self.columns = nn.ModuleList([])

        self.loss = torch.nn.CrossEntropyLoss()
        self.device = None

    def forward(self, observations):
        assert self.columns, 'PNN should at least have one column (missing call to `new_task` ?)'
        x = observations.x
        x = torch.flatten(x, start_dim=1)

        inputs = [c[0](x) for c in self.columns]
        for l in range(1, self.n_layers):
            outputs = []

            for i, column in enumerate(self.columns):
                outputs.append(column[l](inputs[:i+1]))

            inputs = outputs

        y = None
        for t in set(observations.task_labels.tolist()):
            mask = (observations.task_labels == t)
            if y is None:
                y = inputs[t]
            else:
                y[mask] = inputs[t][mask]
        # assert False, print(inputs.size())
        return y


    def new_task(self, sizes, device):
        msg = "Should have the out size for each layer + input size (got {} sizes but {} layers)."
        assert len(sizes) == self.n_layers + 1, msg.format(len(sizes), self.n_layers)
        task_id = len(self.columns)

        modules = []
        for i in range(0, self.n_layers):
            modules.append(PNNLinearBlock(task_id, i, sizes[i], sizes[i+1]))

        new_column = nn.ModuleList(modules).to(device)
        self.columns.append(new_column)
        self.device = device

        print("Add column of the new task")

    def freeze_columns(self, skip=None):
        if skip == None:
            skip = []

        for i, c in enumerate(self.columns):
            if i not in skip:
                for params in c.parameters():
                    params.requires_grad = False

        print("Freeze columns from previous tasks")

    def shared_step(self, batch: Tuple[Observations, Rewards], *args, **kwargs):
        # Since we're training on a Passive environment, we get both
        # observations and rewards.
        observations: Observations = batch[0].to(self.device)
        rewards: Rewards = batch[1]
        image_labels = rewards.y.to(self.device)

        # Get the predictions:
        logits = self(observations)
        y_pred = logits.argmax(-1)
        # print(logits.size())
        loss = self.loss(logits, image_labels)

        accuracy = (y_pred == image_labels).sum().float() / len(image_labels)
        metrics_dict = {"accuracy": accuracy}
        return loss, metrics_dict

    def parameters(self):
        return self.columns[-1].parameters()

class ImproveMethod(Method, target_setting=ClassIncrementalSetting):
    """ 
    Here we implement the method according to the characteristics and methodology of the current proposal. 
    It should be as much as possible agnostic to the model and setting we are going to use. 
    
    The method proposed can be specific to a setting to make comparisons easier. 
    Here what we control is the model's training process, given a setting that delivers data in a certain way.
    """

    @dataclass
    class HParams:
        """ Hyper-parameters of the Settings. """
        # Learning rate of the optimizer.
        learning_rate: float = 0.0001
        n_layers: int = 2
        batch_size: int = 32
        
        @classmethod
        def from_args(cls) -> "HParams":
            """ Get the hparams of the method from the command-line. """
            from simple_parsing import ArgumentParser
            parser = ArgumentParser(description=cls.__doc__)
            parser.add_arguments(cls, dest="hparams")
            args, _ = parser.parse_known_args()
            return args.hparams

    def __init__(self, hparams: HParams = None):
        self.hparams: ImproveMethod.HParams = hparams or self.HParams.from_args()
        self.max_epochs: int = 2

        # print(self.hparams)
        
        # We will create those when `configure` will be called, before training.
        self.model: PNN
        self.optimizer: torch.optim.Optimizer
        self.config: Optional[Config] = None

    def configure(self, setting: ClassIncrementalSetting):
        """ Called before the method is applied on a setting (before training). 

        You can use this to instantiate your model, for instance, since this is
        where you get access to the observation & action spaces.
        """
        # assert False, setting.observation_space.device
        # setting.batch_size = self.hparams.batch_size

        self.layer_size = [np.prod(setting.observation_space[0].shape), 256, setting.increment]
        setting.batch_size = 32

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = PNN(
            n_layers=self.hparams.n_layers,
        )

    def set_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)

    def fit(self, train_env: PassiveEnvironment, valid_env: PassiveEnvironment):
        observations: ClassIncrementalSetting.Observations = train_env.reset()
        cuda_observations = observations.to(device="cuda:0")       
        
        self.model.freeze_columns()
        self.model.new_task(self.layer_size, self.device)
        self.set_optimizer()

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
                    loss, metrics_dict = self.model.shared_step(batch)
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
                    batch_val_loss, metrics_dict = self.model.shared_step(batch)
                    epoch_val_loss += batch_val_loss
                    postfix.update(metrics_dict, val_loss=epoch_val_loss)
                    val_pbar.set_postfix(postfix)
            torch.set_grad_enabled(True)

    def get_actions(self, observations: Observations, action_space: gym.Space) -> Actions:
        """ Get a batch of predictions (aka actions) for these observations. """ 
        with torch.no_grad():
            logits = self.model(observations.to(self.device))
        # Get the predicted classes
        y_pred = logits.argmax(dim=-1)
        return self.target_setting.Actions(y_pred)

    def on_task_switch(self, task_id: Optional[int]):
        # This method gets called if task boundaries are known in the current
        # setting. Furthermore, if task labels are available, task_id will be
        # the index of the new task. If not, task_id will be None.
        # For example, you could do something like this:
        self.model.current_task = task_id

    @classmethod
    def add_argparse_args(cls, parser: ArgumentParser, dest: str = None) -> None:
        parser.add_arguments(cls.HParams, dest="hparams")
        parser.add_argument("--foo", default=123)

    @classmethod
    def from_argparse_args(cls, args: Namespace, dest: str = None) -> "ImproveMethod":
        hparams: ImproveMethod.HParams = args.hparams
        foo: int = args.foo
        method = cls(hparams=hparams)
        return method


def main_command_line():
    from sequoia.settings import TaskIncrementalSetting
    parser = ArgumentParser(description=__doc__, add_dest_to_option_strings=False)
    
    # Add arguments for the Setting
    
    parser.add_arguments(TaskIncrementalSetting, dest="setting")
    # TaskIncrementalSetting.add_argparse_args(parser, dest="setting")
    Config.add_argparse_args(parser, dest="config")
    
    # Add arguments for the Method:
    ImproveMethod.add_argparse_args(parser, dest="method")
    
    args = parser.parse_args()

    # setting: TaskIncrementalSetting = args.setting
    setting: TaskIncrementalSetting = TaskIncrementalSetting.from_argparse_args(args, dest="setting")
    config: Config = Config.from_argparse_args(args, dest="config")

    method: ImproveMethod = ImproveMethod.from_argparse_args(args, dest="method")
    
    method.config = config
    
    
    results = setting.apply(method, config=config)
    print(results.summary())


def main():
    
    # Example: Evaluate a Method on a single CL setting:
    # from sequoia.settings import TaskIncrementalSetting # For Supervised Learning (SL)
    # from sequoia.settings import TaskIncrementalRLSetting # For Reinforcment Learning (RL)

    # Stages:
    ## 1. Creating the setting:
    setting = TaskIncrementalSetting(dataset="mnist", nb_tasks=5)
    # Second option: create the setting from the command-line:
    # setting = TaskIncrementalSetting.from_args()
    
    ## 2. Creating the Method
    method = ImproveMethod()
    
    ## 3. Applying the method to the setting:
    results = setting.apply(method)
    
    print(results.summary())
    print(f"objective: {results.objective}")
    
    exit()

if __name__ == "__main__":
    main_command_line()
    # main()
