""" Patch for the multi-task models in Avalanche, so that we can evaluate on future
tasks, by selecting random prediction.
"""
from typing import Optional

import torch
from torch import Tensor

from avalanche.models import MTSimpleCNN as _MTSimpleCNN
from avalanche.models import MTSimpleMLP as _MTSimpleMLP
from avalanche.models import MultiHeadClassifier as _MultiHeadClassifier
from avalanche.models import SimpleCNN, SimpleMLP


class MultiHeadClassifier(_MultiHeadClassifier):
    def __init__(self, in_features: int, initial_out_features: int = 2):
        """ Multi-head classifier with separate classifiers for each task.

        Typically used in task-incremental scenarios where task labels are
        available and provided to the model.

        :param in_features: number of input features.
        :param initial_out_features: initial number of classes (can be
            dynamically expanded).
        """
        super().__init__(
            in_features=in_features, initial_out_features=initial_out_features
        )

    def forward(self, x: Tensor, task_labels: Optional[Tensor]) -> Tensor:
        if task_labels is None:
            # TODO: Use a task-inference module when `task_labels` is None.
            task_labels = torch.as_tensor([-1 for _ in x], dtype=int)
        return super().forward(x, task_labels)

    def forward_single_task(self, x: Tensor, task_label: Optional[Tensor]):
        """ compute the output given the input `x`. This module uses the task
        label to activate the correct head.

        :param x:
        :param task_label:
        :return:
        """
        if str(task_label) not in self.classifiers:
            # TODO: Let's use the most 'recent' output head instead?
            known_task_labels = list(self.classifiers.keys())
            assert known_task_labels, "Need to have seen at least one task!"
            last_known_task = known_task_labels[-1]
            task_label = last_known_task
            # raise NotImplementedError(
            #     f"Don't yet have an output layer for task {task_label}."
            # )
        return super().forward_single_task(x, task_label)


class MTSimpleCNN(_MTSimpleCNN):
    def __init__(self):
        super().__init__()
        self.classifier = MultiHeadClassifier(in_features=64)

    def forward(self, x: Tensor, task_labels: Optional[Tensor] = None) -> Tensor:
        if task_labels is None:
            # TODO: Use a task-inference module when `task_labels` is None.
            task_labels = torch.as_tensor([-1 for _ in x], dtype=int)
        return super().forward(x=x, task_labels=task_labels)


class MTSimpleMLP(_MTSimpleMLP):
    def __init__(self, input_size: int = 28 * 28, hidden_size: int = 512):
        """
            Multi-task MLP with multi-head classifier.
        """
        super().__init__(input_size=input_size, hidden_size=hidden_size)
        self.classifier = MultiHeadClassifier(in_features=hidden_size)

    def forward(self, x: Tensor, task_labels: Optional[Tensor] = None) -> Tensor:
        if task_labels is None:
            # TODO: Use a task-inference module when `task_labels` is None.
            task_labels = torch.as_tensor([-1 for _ in x], dtype=int)
        return super().forward(x=x, task_labels=task_labels)
