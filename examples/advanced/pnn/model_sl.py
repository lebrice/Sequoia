from typing import Dict, Optional, Tuple, List
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from sequoia.settings.passive.cl.objects import Observations, Rewards
from sequoia.settings import PassiveEnvironment, Actions
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

    def __init__(self, n_layers, arch = 'mlp'):
        super().__init__()
        self.n_layers = n_layers
        self.columns = nn.ModuleList([])

        self.loss = torch.nn.CrossEntropyLoss()
        self.device = None
        self.n_tasks = 0
        self.n_classes_per_task: List[int] = []

        self.arch = arch

    def forward(self, observations):
        assert (
            self.columns
        ), "PNN should at least have one column (missing call to `new_task` ?)"
        x = observations.x

        if self.arch == 'mlp':
            x = torch.flatten(x, start_dim=1)
            # TODO: Debug this:
            #inputs = [
            #    c[0](x) + n_classes_in_task
            #    for n_classes_in_task, c in zip(self.n_classes_per_task, self.columns)
            #]
            inputs = [ c[0](x) for c in self.columns ]

            for l in range(1, self.n_layers):
                outputs = []
                for i, column in enumerate(self.columns):
                    outputs.append(column[l](inputs[: i + 1]))

                inputs = outputs

        else:
            inputs = [ c[0](x) for c in self.columns ]

            outputs = []
            for i, column in enumerate(self.columns):
                outputs.append(column[1](inputs[: i + 1]))

            outputs = [ c[2](outputs[i]) for i,c in enumerate(self.columns) ]
            inputs = [ c[3](outputs[i]) for i,c in enumerate(self.columns) ]


        labels = observations.task_labels
        y: Optional[Tensor] = None
        task_masks = {}
        for task_id in set(labels.tolist()):
            if task_id not in self.added_tasks:
                task_id = random.sample(self.added_tasks, k=1)[0]
                
            task_mask = labels == task_id
            task_masks[task_id] = task_mask

            if y is None:
                y = inputs[task_id]
            else:
                y[task_mask] = inputs[task_id][task_mask]

        assert y is not None, "Can't get prediction in model PNN"
        return y

    # def new_task(self, device, num_inputs, num_actions = 5):
    def new_task(self, device, sizes: List[int]):
        assert len(sizes) == self.n_layers + 1, (
            f"Should have the out size for each layer + input size (got {len(sizes)} "
            f"sizes but {self.n_layers} layers)."
        )
        self.n_tasks += 1
        task_id = len(self.columns)
        # TODO: Fix this to use the actual number of classes per task.
        #self.n_classes_per_task.append(2)
        
        if self.arch == 'conv':
            modules_conv = nn.Sequential()

            modules_conv.add_module('Conv1', PNNConvLayer(task_id, 0, sizes[0], sizes[1], 5))
            modules_conv.add_module('Conv2', PNNConvLayer(task_id, 1, sizes[1], sizes[2], 5))
            #modules_conv.add_module('globavgpool2d', nn.AdaptiveAvgPool2d((1,1)))
            modules_conv.add_module('Flatten', nn.Flatten())
            # modules_conv.add_module('Linear', PNNLinearBlock(task_id, 3, sizes[3],sizes[4]))
            modules_conv.add_module('clf',nn.Linear(sizes[3],sizes[4]))
            self.columns.append(modules_conv.to(device))

        else:
            modules = []
            for i in range(0, self.n_layers):
                modules.append(
                    PNNLinearBlock(col=task_id, depth=i, n_in=sizes[i], n_out=sizes[i + 1])
                )

            new_column = nn.ModuleList(modules).to(device)
            self.columns.append(new_column)
        self.device = device

        # self.encoder.to(device)

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

    def shared_step(
        self,
        batch: Tuple[Observations, Optional[Rewards]],
        environment: PassiveEnvironment,
    ):
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
            rewards when they aren't already part of the batch (as mentioned above).

        Returns
        -------
        Tuple[Tensor, Dict]
            The Loss tensor, and a dict of metrics to be logged.
        """
        # Since we're training on a Passive environment, we will get both observations
        # and rewards, unless we're being evaluated based on our training performance,
        # in which case we will need to send actions to the environments before we can
        # get the corresponding rewards (image labels).
        observations: Observations = batch[0].to(self.device)
        rewards: Optional[Rewards] = batch[1]

        # Get the predictions:
        logits = self(observations)
        y_pred = logits.argmax(-1)
        # TODO: PNN is coded for the DomainIncrementalSetting, where the action space
        # is the same for each task.

        # Get the rewards, if necessary:
        if rewards is None:
            rewards = environment.send(Actions(y_pred))

        image_labels = rewards.y.to(self.device)
        # print(logits.size())
        loss = self.loss(logits, image_labels)

        accuracy = (y_pred == image_labels).sum().float() / len(image_labels)
        metrics_dict = {"accuracy": accuracy.item()}
        return loss, metrics_dict

    def parameters(self, task_id):
        params = []
        for p in self.columns[task_id].parameters():
            params.append(p)

        return params 
