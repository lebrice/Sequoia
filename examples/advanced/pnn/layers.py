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
            self.u.extend([nn.Conv2d(n_in, n_out, kernel_size,
                                     stride=2, padding=1) for _ in range(col)])

    def forward(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]

        cur_column_out = self.layer(inputs[-1])
        prev_columns_out = [mod(x) for mod, x in zip(self.u, inputs)]

        return F.relu(cur_column_out + sum(prev_columns_out))


class PNNGruLayer(nn.Module):
    def __init__(self, col, depth, n_in, n_out):
        super(PNNGruLayer, self).__init__()
        self.layer = nn.GRUCell(n_in, n_out)  # nn.GRUCell(32 * 5 * 5, 256)

        #self.u = nn.ModuleList()
        # if depth > 0:
        #    self.u.extend([ nn.GRUCell(n_in, n_out) for _ in range(col) ])

    def forward(self, inputs, hx):
        # if not isinstance(inputs, list):
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
