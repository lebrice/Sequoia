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
from sequoia.common.spaces import Image

from sequoia.settings import (Method, TaskIncrementalSetting)
from sequoia.settings.passive.cl.objects import (Actions, Observations,
                                                 PassiveEnvironment, Rewards)

from layers import PNNConvLayer, PNNLinearBlock


class PnnClassifier(nn.Module):
    """
    @article{rusu2016progressive,
      title={Progressive neural networks},
      author={Rusu, Andrei A and Rabinowitz, Neil C and Desjardins, Guillaume and Soyer, Hubert and Kirkpatrick, James and Kavukcuoglu, Koray and Pascanu, Razvan and Hadsell, Raia},
      journal={arXiv preprint arXiv:1606.04671},
      year={2016}
    }
    """

    def __init__(self, n_layers):
        super().__init__()
        self.n_layers = n_layers
        self.columns = nn.ModuleList([])

        self.loss = torch.nn.CrossEntropyLoss()
        self.device = None

    def forward(self, observations):
        assert self.columns, 'PNN should at least have one column (missing call to `new_task` ?)'
        x = observations.x
        x = torch.flatten(x, start_dim=1)
        labels = observations.task_labels

        inputs = [c[0](x) for c in self.columns]
        for l in range(1, self.n_layers):
            outputs = []

            for i, column in enumerate(self.columns):
                outputs.append(column[l](inputs[:i+1]))

            inputs = outputs

        y: Optional[Tensor] = None
        task_masks = {}
        for task_id in set(labels.tolist()):
            task_mask = (labels == task_id)
            task_masks[task_id] = task_mask

            if y is None:
                y = inputs[task_id]
            else:
                y[task_mask] = inputs[task_id][task_mask]

        assert y is not None, "Can't get prediction in model PNN"
        return y

    # def new_task(self, device, num_inputs, num_actions = 5):
    def new_task(self, device, sizes):
        assert len(sizes) == self.n_layers + 1, (
            f"Should have the out size for each layer + input size (got {len(sizes)} "
            f"sizes but {self.n_layers} layers)."
        )
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
            for params in c.parameters():
                params.requires_grad = True

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

    def parameters(self, task_id):
        return self.columns[task_id].parameters()
