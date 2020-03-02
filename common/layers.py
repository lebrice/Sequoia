import torch
from torch import nn, Tensor

from typing import *

class Flatten(nn.Module):
    def forward(self, inputs):
        return inputs.view([inputs.shape[0], -1])