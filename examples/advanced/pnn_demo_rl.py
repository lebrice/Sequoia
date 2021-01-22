import sys
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from numpy import inf
import tqdm
import gym
from gym import spaces
from simple_parsing import ArgumentParser

from scipy.signal import lfilter

from sequoia.common import Config
from sequoia.settings import TaskIncrementalRLSetting
from stable_baselines3.common.base_class import BaseAlgorithm
from sequoia.settings import Method

import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class PNNConvLayer(nn.Module):
    def __init__(self, col, depth, n_in, n_out, kernel_size=3):
        super(PNNConvLayer, self).__init__()
        self.col = col
        self.layer = nn.Conv2d(n_in, n_out, kernel_size, stride=2, padding=1)

        self.u = nn.ModuleList()
        if depth > 0:
            self.u.extend([ nn.Conv2d(n_in, n_out, kernel_size, stride=2, padding=1) for _ in range(col) ])

    def forward(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]

        cur_column_out = self.layer(inputs[-1])
        prev_columns_out = [mod(x) for mod, x in zip(self.u, inputs)]

        return F.relu(cur_column_out + sum(prev_columns_out))

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
    def __init__(self, arch='mlp', hidden_size=256):
        super(PNN, self).__init__()
        self.columns_actor = nn.ModuleList([])
        self.columns_critic = nn.ModuleList([])
        self.columns_conv = nn.ModuleList([])
        self.arch = arch
        self.hidden_size = hidden_size

        # Original size 3 x 400 x 600
        self.transformation = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224), 
                                        transforms.ToTensor(), 
                                        ])

    def forward(self, observations):
        assert self.columns_actor, 'PNN should at least have one column (missing call to `new_task` ?)'
        t = observations.task_labels

        if self.arch == 'mlp':
            x = torch.from_numpy(observations.x).unsqueeze(0).float()
            inputs_critic = [c[1](c[0](x)) for c in self.columns_critic]
            inputs_actor = [c[1](c[0](x)) for c in self.columns_actor]

            outputs_critic = []
            outputs_actor = []
            for i, column in enumerate(self.columns_critic):
                outputs_critic.append(column[2](inputs_critic[:i+1]))
                outputs_actor.append(self.columns_actor[i][2](inputs_actor[:i+1]))

            ind_depth = 3

        else:
            x = self.transfor_img(observations.x).unsqueeze(0).float()
            inputs = [c[1](c[0](x)) for c in self.columns_conv]

            outputs = []
            for i, column in enumerate(self.columns_conv):
                outputs.append(column[3](column[2](inputs[:i+1])))

            inputs = outputs
            outputs = []
            for i, column in enumerate(self.columns_conv):
                outputs.append(column[5](column[4](inputs[:i+1])))

            inputs_critic = [ c[6](outputs[i]).view(1,-1) for i,c in enumerate(self.columns_conv) ]
            inputs_actor = inputs_critic[:]

            outputs_critic = []
            outputs_actor = []
            for i, column in enumerate(self.columns_critic):
                outputs_critic.append(column[0](inputs_critic[:i+1]))
                outputs_actor.append(self.columns_actor[i][0](inputs_actor[:i+1]))

            ind_depth = 1

        critic = []
        for i, column in enumerate(self.columns_critic):
            critic.append(column[ind_depth](outputs_critic[i]))

        actor = []
        for i, column in enumerate(self.columns_actor):
            actor.append(F.softmax(column[ind_depth](outputs_actor[i]),dim=1))

        return critic[t], actor[t]

    def new_task(self, device, num_inputs, num_actions = 5):
        task_id = len(self.columns_actor)

        if self.arch == 'conv':
            sizes = [num_inputs, 32, 64, self.hidden_size]
            modules_conv = nn.Sequential()

            modules_conv.add_module('Conv1', PNNConvLayer(task_id, 0, sizes[0], sizes[1]))
            modules_conv.add_module('MaxPool1', nn.MaxPool2d(3))
            modules_conv.add_module('Conv2', PNNConvLayer(task_id, 1, sizes[1], sizes[2]))
            modules_conv.add_module('MaxPool2', nn.MaxPool2d(3))
            modules_conv.add_module('Conv3', PNNConvLayer(task_id, 2, sizes[2], sizes[3]))
            modules_conv.add_module('MaxPool3', nn.MaxPool2d(3))
            modules_conv.add_module('globavgpool2d', nn.AdaptiveAvgPool2d((1,1)))
            self.columns_conv.append(modules_conv)

        modules_actor = nn.Sequential()
        modules_critic = nn.Sequential()

        if self.arch == 'mlp':
            modules_actor.add_module('linAc1',nn.Linear(num_inputs, self.hidden_size))
            modules_actor.add_module('relAc',nn.ReLU(inplace=True))
        modules_actor.add_module('linAc2',PNNLinearBlock(task_id, 1, self.hidden_size, self.hidden_size))
        modules_actor.add_module('linAc3',nn.Linear(self.hidden_size, num_actions))

        if self.arch == 'mlp':
            modules_critic.add_module('linCr1',nn.Linear(num_inputs, self.hidden_size))
            modules_critic.add_module('relCr',nn.ReLU(inplace=True))
        modules_critic.add_module('linCr2',PNNLinearBlock(task_id, 1, self.hidden_size, self.hidden_size))
        modules_critic.add_module('linCr3',nn.Linear(self.hidden_size, 1))

        self.columns_actor.append(modules_actor)
        self.columns_critic.append(modules_critic)

        print("Add column of the new task")

    def freeze_columns(self, skip=None):
        if skip == None:
            skip = []

        for i, c in enumerate(self.columns_actor):
            if i not in skip:
                for params in c.parameters():
                    params.requires_grad = False

                for params in self.columns_critic[i].parameters():
                    params.requires_grad = False

        for i, c in enumerate(self.columns_conv):
            if i not in skip:
                for params in c.parameters():
                    params.requires_grad = False

        print("Freeze columns from previous tasks")

    def parameters(self):
        param = []
        for p in self.columns_critic[-1].parameters():
            param.append(p)
        for p in self.columns_actor[-1].parameters():
            param.append(p)

        if len(self.columns_conv) > 0:
            for p in self.columns_conv[-1].parameters():
                param.append(p)

        return param

    def transfor_img(self, img):
        return self.transformation(img)
        # return lambda img: imresize(img[35:195].mean(2), (80,80)).astype(np.float32).reshape(1,80,80)/255.

class ImproveMethod(Method, target_setting=TaskIncrementalRLSetting):

    @dataclass
    class HParams:
        """ Hyper-parameters of the Settings. """
        # Learning rate of the optimizer.
        learning_rate: float = 2e-4
        num_steps: int = 200
        loss_gamma: float = 0.99
        hidden_size: int = 256

    def __init__(self, hparams: HParams = None):
        self.hparams: ImproveMethod.HParams = hparams or self.HParams.from_args()
        
        # We will create those when `configure` will be called, before training.
        self.model: PNN
        self.optimizer: torch.optim.Optimizer
        self.config: Optional[Config] = None

    def configure(self, setting: TaskIncrementalRLSetting):
        # Delete the model, if present.
        self.model = None
        setting.batch_size = None

        self.num_actions = setting.action_space.n
        image_space: Image = setting.observation_space[0]
        self.num_inputs = np.prod(image_space.shape)

        self.device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

        self.train_steps_per_task = setting.steps_per_task
        self.num_steps = self.hparams.num_steps
        # Otherwise, we can train basically as long as we want on each task.

        self.loss_function = {
            'gamma': self.hparams.loss_gamma,
        }

        if setting.observe_state_directly:
            self.arch = 'mlp'
            self.num_inputs = np.prod(image_space.shape)
        else:
            self.arch = 'conv'
            self.num_inputs = image_space.shape[0]

        self.task_id = 0
        self.hidden_size = self.hparams.hidden_size

    def create_model(self, train_env: gym.Env, valid_env: gym.Env) -> BaseAlgorithm:
        # Create the model, as usual:
        model = PNN(self.arch, self.hidden_size)
        return model

    def on_task_switch(self, task_id: Optional[int]) -> None:
        """ Called when switching tasks in a CL setting. """
        self.task_id = task_id

    def set_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)

    def get_actions(self,
                    observations: TaskIncrementalRLSetting.Observations,
                    action_space: spaces.Space) -> TaskIncrementalRLSetting.Actions:

        predictions = self.model(observations)
        _, logit = predictions
        action = torch.argmax(logit).item()
        return action

    def print_results(self, step, loss, reward):
        print("Task: {}\n Step:{}\n Average Loss: {}\n Average Reward: {}\n".format(self.task_id,
                            step, loss, reward))
    
    def fit(self, train_env: gym.Env, valid_env: gym.Env):
        if self.model is None:
            self.model = self.create_model(train_env, valid_env)

        self.model.freeze_columns()
        self.model.new_task(self.device, self.num_inputs, self.num_actions)
        self.set_optimizer()
        # self.model.float()
        
        all_lengths = []
        average_lengths = []
        all_rewards = []
        entropy_term = 0

        for episode in range(int(self.train_steps_per_task)):
            values = []
            rewards = []
            log_probs = []

            state = train_env.reset()
            for steps in range(self.num_steps):
                value, policy_dist = self.model(state)

                value = value.item() 
                dist = policy_dist.detach().numpy() 

                action = np.random.choice(self.num_actions, p=np.squeeze(dist))
                log_prob = torch.log(policy_dist.squeeze(0)[action])
                entropy = -np.sum(np.mean(dist) * np.log(dist))
                new_state, reward, done, _ = train_env.step(action)

                rewards.append(reward.y)
                values.append(value)
                log_probs.append(log_prob)
                entropy_term += entropy
                state = new_state

                if done or steps == self.num_steps-1:
                    Qval, _ = self.model(state)
                    Qval = Qval.item() 
                    all_rewards.append(np.sum(rewards))
                    all_lengths.append(steps)
                    average_lengths.append(np.mean(all_lengths[-10:]))
                    if episode % 10 == 0:                    
                        sys.stdout.write("episode: {}, reward: {}, total length: {}, average length: {} \n".format(episode, np.sum(rewards), steps, average_lengths[-1]))
                    break

            Qvals = np.zeros_like(values)
            for t in reversed(range(len(rewards))):
                Qval = rewards[t] + self.loss_function['gamma'] * Qval
                Qvals[t] = Qval

            #update actor critic
            values = torch.FloatTensor(values)
            Qvals = torch.FloatTensor(Qvals)
            log_probs = torch.stack(log_probs)

            advantage = Qvals - values
            actor_loss = (-log_probs * advantage).mean()
            critic_loss = 0.5 * advantage.pow(2).mean()
            ac_loss = actor_loss + critic_loss + 0.001 * entropy_term

            self.optimizer.zero_grad()
            ac_loss.backward()
            self.optimizer.step()

    @classmethod
    def add_argparse_args(cls, parser: ArgumentParser, dest: str = None) -> None:
        parser.add_arguments(cls.HParams, dest="hparams")

    @classmethod
    def from_argparse_args(cls, args, dest: str = None) -> "ImproveMethod":
        hparams: ImproveMethod.HParams = args.hparams
        method = cls(hparams=hparams)
        return method

if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__, add_dest_to_option_strings=False)

    Config.add_argparse_args(parser, dest="config")
    ImproveMethod.add_argparse_args(parser, dest="method")

    # Haven't tested with observe_state_directly=False 
    # it run but I don't know if it converge
    setting = TaskIncrementalRLSetting(
        dataset="cartpole",
        observe_state_directly=True,
        nb_tasks=2,
        train_task_schedule={
            0:      {"gravity": 10, "length": 0.3},
            1000:   {"gravity": 10, "length": 0.5},
        },
    )

    args = parser.parse_args()
    
    config: Config = Config.from_argparse_args(args, dest="config")
    method: ImproveMethod = ImproveMethod.from_argparse_args(args, dest="method")
    method.config = config

    ## 2. Creating the Method
    # method = ImproveMethod()
    
    ## 3. Applying the method to the setting:
    results = setting.apply(method, config=config)
    
    print(results.summary())
    print(f"objective: {results.objective}")
    
    exit()