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
    def __init__(self, arch='mlp', hidden_size=256):
        super(PNN, self).__init__()
        self.columns_actor = nn.ModuleList([])
        self.columns_critic = nn.ModuleList([])
        self.arch = arch
        self.hidden_size = hidden_size

    def forward(self, observations, hx):
        assert self.columns_actor, 'PNN should at least have one column (missing call to `new_task` ?)'
        x = torch.from_numpy(observations.x).unsqueeze(0).float()
        t = observations.task_labels

        if self.arch == 'mlp':
            inputs_critic = [c[1](c[0](x)) for c in self.columns_critic]
            inputs_actor = [c[1](c[0](x)) for c in self.columns_actor]

            outputs_critic = []
            outputs_actor = []
            for i, column in enumerate(self.columns_critic):
                outputs_critic.append(column[2](inputs_critic[:i+1]))
                outputs_actor.append(self.columns_actor[i][2](inputs_actor[:i+1]))

            #for l in range(1, 2):
            #    outputs = []
            #    for i, column in enumerate(self.columns):
            #        outputs.append(column[l](inputs[:i+1]))

            ind_depth = 3

        else:
            pass
            # hx = [ hx ]
            # inputs = [c[0](x) for c in self.columns]
            # for l in range(1, 3):
            #     outputs = []
            #     for i, column in enumerate(self.columns):
            #         outputs.append(column[l](inputs[:i+1]))

            # inputs = [ torch.flatten(o, start_dim=1) for o in outputs ]    
            # outputs = []
            # for i, column in enumerate(self.columns):
            #     hx = column[4](inputs[i], hx)
            #     outputs.append(hx)
            # inputs = outputs
            # ind_depth = 5

        critic = []
        for i, column in enumerate(self.columns_critic):
            critic.append(column[ind_depth](outputs_critic[i]))

        actor = []
        for i, column in enumerate(self.columns_actor):
            actor.append(F.softmax(column[ind_depth](outputs_actor[i]),dim=1))

        return critic[t], actor[t], None #outputs[-1]

    def new_task(self, device, num_inputs, num_actions = 5):
        task_id = len(self.columns_actor)

        if self.arch == 'mlp':
            modules_actor = nn.Sequential()
            modules_critic = nn.Sequential()

            modules_actor.add_module('linAc1',nn.Linear(num_inputs, self.hidden_size))
            modules_actor.add_module('relAc',nn.ReLU(inplace=True))
            modules_actor.add_module('linAc2',PNNLinearBlock(task_id, 1, self.hidden_size, self.hidden_size))
            modules_actor.add_module('linAc3',nn.Linear(self.hidden_size, num_actions))

            modules_critic.add_module('linCr1',nn.Linear(num_inputs, self.hidden_size))
            modules_critic.add_module('relCr',nn.ReLU(inplace=True))
            modules_critic.add_module('linCr2',PNNLinearBlock(task_id, 1, self.hidden_size, self.hidden_size))
            modules_critic.add_module('linCr3',nn.Linear(self.hidden_size, 1))

        else:
            pass
            # sizes = [1, 32, 32, 32, 32, 32 * 5 * 5, 256]
            # modules = [ PNNConvLayer(task_id, 0, sizes[0], sizes[1]) ]

            # for depth in range(2,4):
            #     modules.append(PNNConvLayer(task_id, depth, sizes[depth-1], sizes[depth]))

            # modules.append(PNNGruLayer(task_id, 4, sizes[5], sizes[6]))
            # modules.append(PNNLinearBlock(task_id, 5, sizes[6], 1))
            # modules.append(PNNLinearBlock(task_id, 5, sizes[6], num_actions))

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

        print("Freeze columns from previous tasks")

    def parameters(self):
        param = []
        for p in self.columns_critic[-1].parameters():
            param.append(p)
        for p in self.columns_actor[-1].parameters():
            param.append(p)

        return param

    def transfor_img(self, img):
        return lambda img: imresize(img[35:195].mean(2), (80,80)).astype(np.float32).reshape(1,80,80)/255.



# class PNNMethod(Method, target_setting=Setting):
#     def configure(self, setting: Setting):
#         if isinstance(setting, ContinualRLSetting):
#             self.model = PNNA2CModel()
#         else:
#             self.model = PNNResNetClassifier()


class ImproveMethod(Method, target_setting=ContinualRLSetting):

    def configure(self, setting: ContinualRLSetting):
        # Delete the model, if present.
        self.model = None
        setting.batch_size = None

        self.num_actions = setting.action_space.n
        image_space: Image = setting.observation_space[0]
        self.num_inputs = np.prod(image_space.shape)

        self.device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

        self.learning_rate = 3e-4
        self.train_steps_per_task = 500 #setting.max_steps
        self.num_steps = 200
        # Otherwise, we can train basically as long as we want on each task.

        self.loss_function = {
            'gamma': 0.99,
        }

        self.task_id = 0
        self.arch = 'mlp'
        self.hidden_size = 256

    def create_model(self, train_env: gym.Env, valid_env: gym.Env) -> BaseAlgorithm:
        # Create the model, as usual:
        model = PNN(self.arch, self.hidden_size)
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
        action = torch.argmax(logit).item()
        return action

    def print_results(self, step, loss, reward):
        print("Task: {}\n Step:{}\n Average Loss: {}\n Average Reward: {}\n".format(self.task_id,
                            step, loss, reward))
    
    # def fit(self, train_env: gym.Env, valid_env: gym.Env):
    #     if self.model is None:
    #         self.model = self.create_model(train_env, valid_env)

    #     self.model.freeze_columns()
    #     self.model.new_task(self.device, self.num_actions)
    #     self.set_optimizer()
    #     # self.model.float()
    
    
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
            hx = torch.zeros(1, 256)
            for steps in range(self.num_steps):
                # value, policy_dist = actor_critic.forward(state)
                value, policy_dist, hx = self.model(state, hx)

                value = value.item() #detach().numpy()[0,0]
                dist = policy_dist.detach().numpy() 

                action = np.random.choice(self.num_actions, p=np.squeeze(dist))
                log_prob = torch.log(policy_dist.squeeze(0)[action])
                entropy = -np.sum(np.mean(dist) * np.log(dist))
                new_state, reward, done, _ = train_env.step(action)
                # state = train_env.step(action.item())
                # reward = env.reward.y
                # done = env.done.item()
                # env = env.state

                rewards.append(reward.y)
                values.append(value)
                log_probs.append(log_prob)
                entropy_term += entropy
                state = new_state

                if done or steps == self.num_steps-1:
                    Qval, _, _ = self.model(state, hx)
                    Qval = Qval.item() #.detach().numpy()[0,0]
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

if __name__ == "__main__":
    # Stages:
    ## 1. Creating the setting:
    from sequoia.settings import RLSetting
    setting = TaskIncrementalRLSetting(
        dataset="cartpole",
        observe_state_directly=True,
        nb_tasks=2,
        train_task_schedule={
            0:      {"gravity": 10, "length": 0.3},
            1000:   {"gravity": 10, "length": 0.5},
        },
    )
    
    ## 2. Creating the Method
    method = ImproveMethod()
    
    ## 3. Applying the method to the setting:
    results = setting.apply(method)
    
    print(results.summary())
    print(f"objective: {results.objective}")
    
    exit()