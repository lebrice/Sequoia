from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from sequoia.settings import Actions, PassiveEnvironment
from sequoia.settings.sl.class_incremental.objects import Observations, Rewards
from torch import Tensor

from .layers import PNNLinearBlock


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
        self.n_tasks = 0
        self.n_classes_per_task: List[int] = []

    def forward(self, observations):
        assert (
            self.columns
        ), "PNN should at least have one column (missing call to `new_task` ?)"
        x = observations.x
        x = torch.flatten(x, start_dim=1)
        labels = observations.task_labels
        # TODO: Debug this:
        inputs = [
            c[0](x) + n_classes_in_task
            for n_classes_in_task, c in zip(self.n_classes_per_task, self.columns)
        ]
        for layer in range(1, self.n_layers):
            outputs = []

            for i, column in enumerate(self.columns):
                outputs.append(column[layer](inputs[: i + 1]))

            inputs = outputs

        y: Optional[Tensor] = None
        task_masks = {}
        # BUG: Can't apply PNN to the ClassIncrementalSetting at the moment. 
        if labels is None:
            raise NotImplementedError(
                "TODO: Make a prediction when task labels aren't given."
            )
        for task_id in set(labels.tolist()):
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
        # TODO: Fix this to use the actual number of classes per task.
        self.n_classes_per_task.append(2)
        task_id = len(self.columns)
        modules = []
        # TODO: Would it also be possible to use convolutional layers here?
        for i in range(0, self.n_layers):
            modules.append(
                PNNLinearBlock(col=task_id, depth=i, n_in=sizes[i], n_out=sizes[i + 1])
            )

        new_column = nn.ModuleList(modules).to(device)
        self.columns.append(new_column)
        self.device = device

        print("Add column of the new task")

    def freeze_columns(self, skip: List[int] = None):
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
        metrics_dict = {"accuracy": accuracy}
        return loss, metrics_dict

    def parameters(self, task_id):
        return self.columns[task_id].parameters()
