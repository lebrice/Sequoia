""" Example Method for the SL track: Multi-Head Classifier with simple task inference.

You can use this model and method as a jumping off point for your own submission.
"""
from dataclasses import dataclass, replace
from logging import getLogger
from typing import Optional

import torch
from gym import Space, spaces
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer

from sequoia.settings.sl.incremental import ClassIncrementalSetting
from sequoia.settings.sl.incremental.objects import Observations

from .classifier import Classifier, ExampleMethod

logger = getLogger(__file__)


class MultiHeadClassifier(Classifier):
    @dataclass
    class HParams(Classifier.HParams):
        pass

    def __init__(
        self,
        observation_space: Space,
        action_space: spaces.Discrete,
        reward_space: spaces.Discrete,
        hparams: "MultiHeadClassifier.HParams" = None,
    ):
        super().__init__(observation_space, action_space, reward_space, hparams=hparams)
        # Use one output layer per task, rather than a single layer.
        self.output_heads = nn.ModuleList()
        # Use the output layer created in the Classifier constructor for task 0.
        self.output_heads.append(self.output)

        # NOTE: The optimizer will be set here, so that we can add the parameters of any
        # new output heads to it later.
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.current_task_id: int = 0

    def configure_optimizers(self) -> Optimizer:
        self.optimizer = super().configure_optimizers()
        return self.optimizer

    def create_output_head(self) -> nn.Module:
        return nn.Linear(self.representations_size, self.n_classes).to(self.device)

    def get_or_create_output_head(self, task_id: int) -> nn.Module:
        """Retrieves or creates a new output head for the given task index.

        Also stores it in the `output_heads`, and adds its parameters to the
        optimizer.
        """
        task_output_head: nn.Module
        if len(self.output_heads) > task_id:
            task_output_head = self.output_heads[task_id]
        else:
            logger.info(f"Creating a new output head for task {task_id}.")
            task_output_head = self.create_output_head()
            self.output_heads.append(task_output_head)
            assert self.optimizer, "need to set `optimizer` on the model."
            self.optimizer.add_param_group({"params": task_output_head.parameters()})
        return task_output_head

    def forward(self, observations: Observations) -> Tensor:
        """Smart forward pass with multi-head predictions and task inference.

        This forward pass can handle three different scenarios, depending on the
        contents of `observations.task_labels`:
        1.  Base case: task labels are present, and all examples are from the same task.
            - Perform the 'usual' forward pass (e.g. `super().forward(observations)`).
        2.  Task labels are present, and the batch contains a mix of samples from
            different tasks:
            - Create slices of the batch for each task, where all items in each
              'sub-batch' come from the same task.
            - Perform a forward pass for each task, by calling `forward` recursively
              with the sub-batch for each task as an argument (Case 1).
        3.  Task labels are *not* present. Perform some type of task inference, using
            the `task_inference_forward_pass` method. Check its docstring for more info.

        Parameters
        ----------
        observations : Observations
            Observations from an environment. As of right now, all Settings produce
            observations with (at least) the two following attributes:
            - x: Tensor (the images/inputs)
            - task_labels: Optional[Tensor] (The task labels, when available, else None)

        Returns
        -------
        Tensor
            The outputs, which in this case are the classification logits.
            All three cases above produce the same kind of outputs.
        """
        observations = observations.to(self.device)
        task_ids: Optional[Tensor] = observations.task_labels

        if task_ids is None:
            # Run the forward pass with task inference turned on.
            return self.task_inference_forward_pass(observations)

        task_ids_present_in_batch = torch.unique(task_ids)
        if len(task_ids_present_in_batch) > 1:
            # Case 2: The batch contains data from more than one task.
            return self.split_forward_pass(observations)

        # Base case: "Normal" forward pass, where all items come from the same task.
        # - Setup the model for this task, however you want, and then do a forward pass,
        # as you normally would.
        # NOTE: If you want to reuse this cool multi-headed forward pass in your
        # own model, these lines here are what you'd want to change.
        task_id: int = task_ids_present_in_batch.item()

        # <--------------- Change below ---------------->
        if task_id == self.current_task_id:
            output_head = self.output
        else:
            output_head = self.get_or_create_output_head(task_id)
        features = self.encoder(observations.x)
        logits = output_head(features)
        return logits

    def split_forward_pass(self, observations: Observations) -> Tensor:
        """Perform a forward pass for a batch of observations from different tasks.

        This is called in `forward` when there is more than one unique task label in the
        batch.
        This will call `forward` for each task id present in the batch, passing it a
        slice of the batch, in which all items are from that task.

        NOTE: This cannot cause recursion problems, because `forward`(d=2) will be
        called with a bach of items, all of which come from the same task. This makes it
        so `split_forward_pass` cannot then be called again.

        Parameters
        ----------
        observations : Observations
            Observations, in which the task labels might not all be the same.

        Returns
        -------
        Tensor
            The outputs/logits from each task, re-assembled into a single batch, with
            the task ordering from `observations` preserved.
        """
        assert observations.task_labels is not None
        # We have task labels.
        task_labels: Tensor = observations.task_labels
        unique_task_ids, inv_indices = torch.unique(task_labels, return_inverse=True)
        # There might be more than one task in the batch.
        batch_size = observations.batch_size
        all_indices = torch.arange(batch_size, dtype=int, device=self.device)

        # Placeholder for the predicitons for each item in the batch.
        task_outputs = [None for _ in range(batch_size)]

        for i, task_id in enumerate(unique_task_ids):
            # Get the forward pass slice for this task.
            # Boolean 'mask' tensor, that selects entries from task `task_id`.
            is_from_this_task = inv_indices == i
            # Indices of the batch elements that are from task `task_id`.
            task_indices = all_indices[is_from_this_task]

            # Take a slice of the observations, in which all items come from this task.
            task_observations = observations[is_from_this_task]
            # Perform a "normal" forward pass (Base case).
            task_output = self.forward(task_observations)

            # Store the outputs for the items from this task.
            for i, index in enumerate(task_indices):
                task_outputs[index] = task_output[i]

        # Merge the results.
        assert all(item is not None for item in task_outputs)
        logits = torch.stack(task_outputs)
        return logits

    def task_inference_forward_pass(self, observations: Observations) -> Tensor:
        """Forward pass with a simple form of task inference."""
        # We don't have access to task labels (`task_labels` is None).
        # --> Perform a simple kind of task inference:
        # 1. Perform a forward pass with each task's output head;
        # 2. Merge these predictions into a single prediction somehow.
        assert observations.task_labels is None

        # NOTE: This assumes that the observations are batched.
        # These are used below to indicate the shape of the different tensors.
        B = observations.x.shape[0]
        T = n_known_tasks = len(self.output_heads)
        N = self.n_classes
        # Tasks encountered previously and for which we have an output head.
        known_task_ids: list[int] = list(range(n_known_tasks))
        assert known_task_ids
        # Placeholder for the predictions from each output head for each item in the
        # batch
        task_outputs = [None for _ in known_task_ids]  # [T, B, N]

        # Get the forward pass for each task.
        for task_id in known_task_ids:
            # Create 'fake' Observations for this forward pass, with 'fake' task labels.
            # NOTE: We do this so we can call `self.forward` and not get an infinite
            # recursion.
            task_labels = torch.full([B], task_id, device=self.device, dtype=int)
            task_observations = replace(observations, task_labels=task_labels)

            # Setup the model for task `task_id`, and then do a forward pass.
            task_logits = self.forward(task_observations)

            task_outputs[task_id] = task_logits

        # 'Merge' the predictions from each output head using some kind of task
        # inference.
        assert all(item is not None for item in task_outputs)
        # Stack the predictions (logits) from each output head.
        logits_from_each_head: Tensor = torch.stack(task_outputs, dim=1)
        assert logits_from_each_head.shape == (B, T, N)

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
        mask = F.one_hot(most_probable_class, N).to(dtype=bool, device=self.device)
        chosen_output_head_per_item = chosen_head_per_class[mask]
        assert mask.shape == (B, N)
        assert chosen_output_head_per_item.shape == (B,)
        # Example (continued):
        # mask: [[False, True, False, True]]
        # chosen_output_head_per_item: [0]

        # Create a bool tensor to select items associated with the chosen output head.
        selected_mask = F.one_hot(chosen_output_head_per_item, T).to(dtype=bool, device=self.device)
        assert selected_mask.shape == (B, T)
        # Select the logits using the mask:
        logits = logits_from_each_head[selected_mask]
        assert logits.shape == (B, N)
        return logits

    def on_task_switch(self, task_id: Optional[int]):
        """Executed when the task switches (to either a known or unknown task)."""
        if task_id is not None:
            # Switch the output head.
            self.current_task_id = task_id
            self.output = self.get_or_create_output_head(task_id)


class ExampleTaskInferenceMethod(ExampleMethod):
    def __init__(self, hparams: MultiHeadClassifier.HParams = None):
        super().__init__(hparams=hparams or MultiHeadClassifier.HParams())

    def configure(self, setting: ClassIncrementalSetting):
        """Called before the method is applied on a setting (before training).

        You can use this to instantiate your model, for instance, since this is
        where you get access to the observation & action spaces.
        """
        self.model = MultiHeadClassifier(
            observation_space=setting.observation_space,
            action_space=setting.action_space,
            reward_space=setting.reward_space,
            hparams=self.hparams,
        )
        self.optimizer = self.model.configure_optimizers()
        # Share a reference to the Optimizer with the model, so it can add new weights
        # when needed.
        self.model.optimizer = self.optimizer

    def on_task_switch(self, task_id: Optional[int]):
        self.model.on_task_switch(task_id)

    def get_actions(self, observations, action_space):
        return super().get_actions(observations, action_space)


if __name__ == "__main__":
    # Create the Method, either manually:
    # method = ExampleTaskInferenceMethod()
    # Or, from the command-line:
    from simple_parsing import ArgumentParser

    from sequoia.settings.sl.class_incremental import (
        ClassIncrementalSetting,
        TaskIncrementalSLSetting,
    )

    parser = ArgumentParser(description=__doc__)
    ExampleTaskInferenceMethod.add_argparse_args(parser)
    args = parser.parse_args()
    method = ExampleTaskInferenceMethod.from_argparse_args(args)

    # Create the Setting:

    # Simpler Settings (useful for debugging):
    # setting = TaskIncrementalSLSetting(
    # setting = ClassIncrementalSetting(
    #     dataset="mnist",
    #     nb_tasks=5,
    #     monitor_training_performance=True,
    #     batch_size=32,
    #     num_workers=4,
    # )

    # Very similar setup to the SL Track of the competition:
    setting = ClassIncrementalSetting(
        dataset="synbols",
        nb_tasks=12,
        monitor_training_performance=True,
        known_task_boundaries_at_test_time=False,
        batch_size=32,
        num_workers=4,
    )
    results = setting.apply(method)
