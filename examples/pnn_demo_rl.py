import sys
import torch
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from numpy import inf
import tqdm
import gym
from gym import spaces

from scipy.signal import lfilter

from sequoia.settings.active.continual import ContinualRLSetting
from sequoia.settings import TaskIncrementalRLSetting

from stable_baselines3.common.base_class import BaseAlgorithm
from sequoia.settings import Method
from sequoia.methods.stable_baselines3_methods import A2CModel
from sequoia.methods.stable_baselines3_methods import StableBaselines3Method

import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, arch='mlp'):
        super(PNN, self).__init__()
        self.columns = nn.ModuleList([])
        self.loss = torch.nn.CrossEntropyLoss()
        self.device = None
        self.arch = arch

    def forward(self, observations, hx):
        assert self.columns, 'PNN should at least have one column (missing call to `new_task` ?)'
        x = observations.x.unsqueeze(0).float()
        t = observations.task_labels

        if self.arch == 'mlp':
            inputs = [c[0](x) for c in self.columns]
            for l in range(1, 2):
                outputs = []
                for i, column in enumerate(self.columns):
                    outputs.append(column[l](inputs[:i+1]))

                inputs = outputs
            ind_depth = 2

        else:
            hx = [ hx ]
            inputs = [c[0](x) for c in self.columns]
            for l in range(1, 3):
                outputs = []
                for i, column in enumerate(self.columns):
                    outputs.append(column[l](inputs[:i+1]))

            inputs = [ torch.flatten(o, start_dim=1) for o in outputs ]    
            outputs = []
            for i, column in enumerate(self.columns):
                hx = column[4](inputs[i], hx)
                outputs.append(hx)
            inputs = outputs
            ind_depth = 5

        critic = []
        for i, column in enumerate(self.columns):
            critic.append(column[ind_depth](inputs[:i+1]))

        actor = []
        for i, column in enumerate(self.columns):
            actor.append(column[ind_depth+1](inputs[:i+1]))
        
        return critic[t], actor[t], outputs[-1]

    def new_task(self, device, num_actions = 5):
        task_id = len(self.columns)
        if self.arch == 'mlp':
            sizes = [4, 32, 32]
            modules = []
            for i in range(0, 2):
                modules.append(PNNLinearBlock(task_id, i, sizes[i], sizes[i+1]))
            modules.append(PNNLinearBlock(task_id, 2, sizes[2], 1))
            modules.append(PNNLinearBlock(task_id, 2, sizes[2], num_actions))
        else:
            sizes = [1, 32, 32, 32, 32, 32 * 5 * 5, 256]
            modules = [ PNNConvLayer(task_id, 0, sizes[0], sizes[1]) ]

            for depth in range(2,4):
                modules.append(PNNConvLayer(task_id, depth, sizes[depth-1], sizes[depth]))

            modules.append(PNNGruLayer(task_id, 4, sizes[5], sizes[6]))
            modules.append(PNNLinearBlock(task_id, 5, sizes[6], 1))
            modules.append(PNNLinearBlock(task_id, 5, sizes[6], num_actions))

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

    def parameters(self):
        return self.columns[-1].parameters()

    def transfor_img(self, img):
        return lambda img: imresize(img[35:195].mean(2), (80,80)).astype(np.float32).reshape(1,80,80)/255.

class ImproveMethod(Method, target_setting=ContinualRLSetting):

    def configure(self, setting: ContinualRLSetting):
        # Delete the model, if present.
        self.model = None
        setting.batch_size = None
        setting.transforms = []
        setting.train_transforms = []
        setting.val_transforms = []
        setting.test_transforms = []

        self.num_actions = setting.action_space.n
        self.device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

        self.learning_rate = 0.0001
        self.total_timesteps_per_task = setting.steps_per_task
        self.train_steps_per_task = setting.max_steps
        # Otherwise, we can train basically as long as we want on each task.

        self.loss_function = {
            'gamma': 0.99,
            'tau': 1.0,
        }

        self.task_id = 0
        self.arch = 'mlp'

    def create_model(self, train_env: gym.Env, valid_env: gym.Env) -> BaseAlgorithm:
        # Create the model, as usual:
        model = PNN(self.arch)
        self.task_trained = []
        return model

    def on_task_switch(self, task_id: Optional[int]) -> None:
        """ Called when switching tasks in a CL setting. """
        self.task_id = task_id

    def set_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def get_actions(self,
                    observations: ContinualRLSetting.Observations,
                    action_space: spaces.Space) -> ContinualRLSetting.Actions:
        # obs = observations.x
        hx = torch.zeros(1, 256)
        predictions = self.model(observations, hx)
        _, logit, _ = predictions
        logp = F.log_softmax(logit, dim=-1)[0]
        action = torch.argmax(logp)
        # BUG: DQN prediction here doesn't work. 
        if action not in action_space:
            assert len(action) == 1, (observations, action, action_space)
            action = action.item()
        return action

    def cost_func(self, values, logps, actions, rewards):
        discount = lambda x, gamma: lfilter([1],[1,-gamma],x[::-1])[::-1]

        np_values = values.view(-1).data.numpy()

        delta_t = np.asarray(rewards) + self.loss_function['gamma'] * np_values[1:] - np_values[:-1]
        logpys = logps.gather(1, torch.tensor(actions).view(-1,1))
        gen_adv_est = discount(delta_t, self.loss_function['gamma'] * self.loss_function['tau'])
        policy_loss = -(logpys.view(-1) * torch.FloatTensor(gen_adv_est.copy())).sum()
        
        rewards[-1] += self.loss_function['gamma'] * np_values[-1]
        discounted_r = discount(np.asarray(rewards), self.loss_function['gamma'])
        discounted_r = torch.tensor(discounted_r.copy(), dtype=torch.float32)
        value_loss = .5 * (discounted_r - values[:-1,0]).pow(2).sum()

        entropy_loss = -(-logps * torch.exp(logps)).sum()
        return policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

    def print_results(self, step, loss, reward):
        print("Task: {}\n Step:{}\n Average Loss: {}\n Average Reward: {}\n".format(self.task_id,
                            step, loss, reward))

    def fit(self, train_env: gym.Env, valid_env: gym.Env):
        if self.model is None:
            self.model = self.create_model(train_env, valid_env)

        self.model.freeze_columns()
        self.model.new_task(self.device, self.num_actions)
        self.set_optimizer()
        # self.model.float()
        
        task_reward = []
        task_loss = []
        done = True

        total_episodes = 0
        for i in range(int(self.train_steps_per_task/20)):
            values = []
            logps = []
            actions = []
            rewards = []
            for _ in range(20):
                if done:
                    hx = torch.zeros(1, 256)
                    episode_length, episode_reward, episode_loss = 0, 0, 0
                    env = train_env.reset()
                    if self.arch != 'mlp':
                        env.x = torch.tensor(self.model.transfor_img(env.x))
                else:
                    hx = hx.detach()

                episode_length += 1
                value, logit, hx = self.model(env, hx)
                logp = F.log_softmax(logit, dim=-1)

                action = torch.exp(logp).multinomial(num_samples=1)

                env = train_env.step(action.item())
                reward = env.reward.y
                done = env.done.item()
                env = env.state

                if self.arch != 'mlp':
                    env.x = torch.tensor(self.model.transfor_img(env.x))
                episode_reward += reward
                reward = np.clip(reward, -1, 1)
                # done = done or episode_length >= 1e4

                values.append(value.detach())
                logps.append(logp)
                actions.append(action.detach())
                rewards.append(reward)
                
                if total_episodes % 100 == 0:
                    self.print_results(i, np.mean(task_loss), np.mean(task_reward))

                total_episodes += 1

            if done:
                next_value = torch.zeros(1,1)
            else:
                next_value = self.model(env, hx)[0]
            values.append(next_value.detach())

            loss = self.cost_func(torch.cat(values), torch.cat(logps), torch.cat(actions), np.asarray(rewards))
            episode_loss += loss.item()
            task_loss.append(loss.item())
            task_reward.append(reward.item())

            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 40)
            self.optimizer.step()


if __name__ == "__main__":
    # Stages:
    ## 1. Creating the setting:
    setting = TaskIncrementalRLSetting(
        dataset="cartpole",
        observe_state_directly=True,
        nb_tasks=2,
        train_task_schedule={
            0:      {"gravity": 10, "length": 0.3},
            1000:   {"gravity": 10, "length": 0.5},
        },
        max_steps = 2000,
    )
    
    ## 2. Creating the Method
    method = ImproveMethod()
    
    ## 3. Applying the method to the setting:
    results = setting.apply(method)
    
    print(results.summary())
    print(f"objective: {results.objective}")
    
    exit()