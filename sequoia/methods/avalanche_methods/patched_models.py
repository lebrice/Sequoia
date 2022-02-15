""" Patch for the multi-task models in Avalanche, so that we can evaluate on future
tasks, by selecting random prediction.
"""
import warnings
from abc import abstractmethod
from typing import Any, List, Optional

import torch
from avalanche.models import MTSimpleCNN as _MTSimpleCNN
from avalanche.models import MTSimpleMLP as _MTSimpleMLP
from avalanche.models import MultiHeadClassifier as _MultiHeadClassifier
from avalanche.models.dynamic_modules import MultiTaskModule
from torch import Tensor
from torch.nn import functional as F

from sequoia.utils import get_logger

logger = get_logger(__name__)


class PatchedMultiTaskModule(MultiTaskModule):
    @property
    @abstractmethod
    def known_task_ids(self) -> List[Any]:
        pass

    def task_inference_forward_pass(self, x: Tensor) -> Tensor:
        """Forward pass with a simple form of task inference."""
        # We don't have access to task labels (`task_labels` is None).
        # --> Perform a simple kind of task inference:
        # 1. Perform a forward pass with each task's output head;
        # 2. Merge these predictions into a single prediction somehow.

        # NOTE: This assumes that the observations are batched.
        # These are used below to indicate the shape of the different tensors.
        B = x.shape[0]
        T = len(self.known_task_ids)
        # N = self.action_space.n
        # Tasks encountered previously and for which we have an output head.
        # TODO: This assumes that the keys of the ModuleDict are integers.
        known_task_ids: List[int] = list(int(t) for t in self.known_task_ids)
        assert known_task_ids
        # Placeholder for the predictions from each output head for each item in the
        # batch
        task_outputs = [None for _ in known_task_ids]  # [T, B, N]

        # Get the forward pass for each task.
        for task_id in known_task_ids:
            # Create 'fake' Observations for this forward pass, with 'fake' task labels.
            # NOTE: We do this so we can call `self.forward` and not get an infinite
            # recursion.
            task_labels = torch.full([B], task_id, device=x.device, dtype=int)
            # task_observations = replace(observations, task_labels=task_labels)

            # Setup the model for task `task_id`, and then do a forward pass.
            task_forward_pass = self.forward(x, task_labels=task_labels)

            task_outputs[task_id] = task_forward_pass
        if len(task_outputs) == 1:
            return task_outputs[0]

        N = max(task_output.shape[-1] for task_output in task_outputs)

        # 'Merge' the predictions from each output head using some kind of task
        # inference.
        assert all(item is not None for item in task_outputs)
        # Stack the predictions (logits) from each output head.
        # NOTE: Here in Avalanche it's possible that each output head's output had a
        # different shape. Therefore we need to handle it like a list of tensors rather
        # than a stacked tensor.
        if all(not task_output.shape[-1] == N for task_output in task_outputs):
            raise NotImplementedError("TODO: Output heads didn't give outputs of the same shape!")
            # logits_from_each_head = task_outputs
            # probs_from_each_head = [
            #     torch.softmax(head_logits, dim=-1) for head_logits in logits_from_each_head
            # ]
            # IDEA: Add zeros to the outputs of a different shape.
        else:
            logits_from_each_head = torch.stack(task_outputs, dim=1)
            # Normalize the logits from each output head with softmax.
            # Example with batch size of 1, output heads = 2, and classes = 4:
            # logits from each head:  [[[123, 456, 123, 123], [1, 1, 2, 1]]]
            # 'probs' from each head: [[[0.1, 0.6, 0.1, 0.1], [0.2, 0.2, 0.4, 0.2]]]
            probs_from_each_head = torch.softmax(logits_from_each_head, dim=-1)

        assert probs_from_each_head.shape == (B, T, N)
        # Simple kind of task inference:
        # For each item in the batch, use the class that has the highest probability
        # accross all output heads.
        max_probs_across_heads, chosen_head_per_class = probs_from_each_head.max(dim=1)
        assert max_probs_across_heads.shape == (B, N)
        assert chosen_head_per_class.shape == (B, N)
        # Example (continued):
        # max probs across heads:        [[0.2, 0.6, 0.4, 0.2]]
        # chosen output heads per class: [[1, 0, 1, 1]]

        # Determine which output head has highest "confidence":
        max_prob_value, most_probable_class = max_probs_across_heads.max(dim=1)
        assert max_prob_value.shape == (B,)
        assert most_probable_class.shape == (B,)
        # Example (continued):
        # max_prob_value: [0.6]
        # max_prob_class: [1]

        # A bit of boolean trickery to get what we need, which is, for each item, the
        # index of the output head that gave the most confident prediction.
        mask = F.one_hot(most_probable_class, N).to(dtype=bool, device=x.device)
        chosen_output_head_per_item = chosen_head_per_class[mask]
        assert mask.shape == (B, N)
        assert chosen_output_head_per_item.shape == (B,)
        # Example (continued):
        # mask: [[False, True, False, True]]
        # chosen_output_head_per_item: [0]

        # Create a bool tensor to select items associated with the chosen output head.
        selected_mask = F.one_hot(chosen_output_head_per_item, T).to(dtype=bool, device=x.device)
        assert selected_mask.shape == (B, T)
        # Select the logits using the mask:
        selected_outputs = logits_from_each_head[selected_mask]
        assert selected_outputs.shape == (B, N)
        return selected_outputs


from avalanche.benchmarks.utils import AvalancheDataset


class MultiHeadClassifier(_MultiHeadClassifier):
    def __init__(self, in_features: int, initial_out_features: int = 2):
        """Multi-head classifier with separate classifiers for each task.

        Typically used in task-incremental scenarios where task labels are
        available and provided to the model.

        :param in_features: number of input features.
        :param initial_out_features: initial number of classes (can be
            dynamically expanded).
        """
        super().__init__(in_features=in_features, initial_out_features=initial_out_features)

    def adaptation(self, dataset: AvalancheDataset):
        """If `dataset` contains new tasks, a new head is initialized.

        :param dataset: data from the current experience.
        :return:
        """
        super().adaptation(dataset)

    def forward(self, x: Tensor, task_labels: Optional[Tensor]) -> Tensor:
        if task_labels is None:
            # We don't do task inference in this layer, since it's handled in the
            # patched models below.
            raise NotImplementedError("Shouldn't get None task labels in the MultiHeadClassifier!")
        else:
            assert isinstance(task_labels, Tensor)
        return super().forward(x, task_labels)

    def forward_single_task(self, x: Tensor, task_label: Optional[Tensor]):
        """compute the output given the input `x`. This module uses the task
        label to activate the correct head.

        :param x:
        :param task_label:
        :return:
        """
        if task_label is not None:
            if not isinstance(task_label, int):
                task_label = task_label.item()
        # TODO: If/when we make the context variable truly continuous, then this
        # won't work.
        assert task_label is None or isinstance(task_label, int), task_label

        if str(task_label) not in self.classifiers:
            # TODO: Let's use the most 'recent' output head instead?
            known_task_labels = list(self.classifiers.keys())
            assert known_task_labels, "Need to have seen at least one task!"
            last_known_task = known_task_labels[-1]
            task_label = last_known_task
            warnings.warn(
                RuntimeWarning(
                    f"performing forward pass on previously unseen task, will pretend "
                    f"inputs come from task {last_known_task} instead."
                )
            )
        return super().forward_single_task(x, task_label)


class MTSimpleCNN(_MTSimpleCNN, PatchedMultiTaskModule):
    def __init__(self):
        super().__init__()
        self.classifier = MultiHeadClassifier(in_features=64)

    def forward(self, x: Tensor, task_labels: Optional[Tensor] = None) -> Tensor:
        if task_labels is None:
            # NOTE: When training, we could rely on a property like `current_task_id`
            # being set within the `on_task_switch` callback.
            # The reason for this is that in some of the strategies, `GEM` strategy (and
            # others), when training they sometimes don't pass a task index! In the case
            # of GEM though, it doesnt pass the task id when calculating the
            # reference gradient, so I'm not sure we want to be using this in this case.
            if self.training:
                warnings.warn(
                    RuntimeWarning("Using task inference in the forward pass while training?")
                )
            return self.task_inference_forward_pass(x=x)
        return super().forward(x=x, task_labels=task_labels)

    @property
    def known_task_ids(self) -> List[Any]:
        return list(self.classifier.classifiers.keys())


class MTSimpleMLP(_MTSimpleMLP, PatchedMultiTaskModule):
    def __init__(self, input_size: int = 28 * 28, hidden_size: int = 512):
        """
        Multi-task MLP with multi-head classifier.
        """
        super().__init__(input_size=input_size, hidden_size=hidden_size)
        self.classifier = MultiHeadClassifier(in_features=hidden_size)

    def forward(self, x: Tensor, task_labels: Optional[Tensor] = None) -> Tensor:
        if task_labels is None:
            if self.training:
                warnings.warn(
                    RuntimeWarning("Using task inference in the forward pass while training?")
                )
            return self.task_inference_forward_pass(x=x)
        return super().forward(x=x, task_labels=task_labels)

    @property
    def known_task_ids(self) -> List[Any]:
        return list(self.classifier.classifiers.keys())
