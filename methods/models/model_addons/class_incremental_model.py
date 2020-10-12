from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union, Set, Sequence
from contextlib import contextmanager

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from pytorch_lightning.core.decorators import auto_move_data

from common.config import Config

from settings import ClassIncrementalSetting, Observations, Actions, Rewards
from utils import dict_intersection, zip_dicts, prod
from utils.logging_utils import get_logger

from .semi_supervised_model import SemiSupervisedModel
from ..base_model import Batch, BaseModel, OutputHead
logger = get_logger(__file__)


SettingType = TypeVar("SettingType", bound=ClassIncrementalSetting)


class ClassIncrementalModel(BaseModel[SettingType]):
    """ Extension of the Model LightningModule aimed at CL settings.
    TODO: Add the stuff related to multihead/continual learning here?
    """

    @dataclass
    class HParams(BaseModel.HParams):
        """ Hyperparameters specific to a Continual Learning classifier.
        TODO: Add any hyperparameters specific to CL here.
        """
        # Wether to create one output head per task.
        # TODO: Does it make no sense to have multihead=True when the model doesn't
        # have access to task labels. Need to figure out how to manage this between TaskIncremental and Classifier.
        multihead: bool = False

    def __init__(self, setting: ClassIncrementalSetting, hparams: HParams, config: Config):
        self._output_head: OutputHead = None

        super().__init__(setting=setting, hparams=hparams, config=config)
        self.hp: ClassIncrementalModel.HParams
        self.setting: SettingType

        # TODO: Add an optional task inference mechanism for ClassIncremental
        # methods!
        self.task_inference_module: nn.Module = None
        
        self.previous_task: Optional[int] = None
        self.current_task: Optional[int] = None

        self.output_heads: Dict[str, OutputHead] = nn.ModuleDict()
        if self.hp.multihead:
            output_head = self.create_output_head()
            self.output_head = output_head
            self.output_heads[str(self.setting.current_task_id)] = output_head
    
    @property
    def output_head(self) -> OutputHead:
        """ Get the output head for the current task.

        FIXME: It's generally bad practice to do heavy computation on a property
        so we should probably add something like a get_output_head(task) method.
        """
        if self.hp.multihead:
            if ((self.training and self.setting.task_labels_at_train_time) or
                (not self.training and self.setting.task_labels_at_test_time)):
                current_task_id = self.current_task
                # current_task_id = self.setting.current_task_id

            elif self.task_inference_module is not None:
                # current_task_id = self.task_inference_module(...)
                raise NotImplementedError("TODO")
            
            # TODO: Look into this, seems a bit weird.
            elif self._output_head is not None:
                # Just return the current output head.
                return self._output_head
            else:
                raise RuntimeError("No way of determining the task id and output head is None!")

            key = str(current_task_id)
            if key not in self.output_heads:
                # Create the output head, since it's not already in there.
                output_head = self.create_output_head()
                self.output_heads[key] = output_head
            else:
                output_head = self.output_heads[key]
            self._output_head = output_head
            # Return the output head for the current task.
            return output_head
        if self._output_head is None:
            self._output_head = self.create_output_head()
        return self._output_head

    @output_head.setter
    def output_head(self, value: OutputHead) -> None:
        # logger.debug(f"Setting output head to {value}")
        self._output_head = value

    # @auto_move_data
    def forward(self, input_batch: Any) -> Dict[str, Tensor]:
        """ Forward pass of the Model. Returns a dict."""
        # Just testing things out here.
        observation: Observations = self.Observations.from_inputs(input_batch)
        assert isinstance(observation, self.Observations)
        
        # Get the task labels from the observation.
        task_labels = observation.task_labels
        
        # IDEA: This would basically call super.forward() on the slices of the
        # batch, and then re-combine the forward pass dicts before returning
        # the results.
        # It's a bit extra. Maybe we only really ever want to have the output
        # task be the 'branched-out/multi-task' portion.
        if task_labels is None or not len(task_labels):
            # Default back to the behaviour of the parent class, which will use
            # the current output head (at attribute `self.output_head`).
            return super().forward(observation)

        if isinstance(task_labels, (Tensor, np.ndarray)):
            unique_task_labels = torch.unique(task_labels).tolist()
        else:
            unique_task_labels = list(set(task_labels))

        if len(unique_task_labels) == 1:
            # If everything is in the same task, no need to split/merge.
            task_id = unique_task_labels[0]
            with self.temporarily_in_task(task_id):
                return super().forward(observation)

        batch_size = observation.batch_size

        # The 'merged' forward pass result dict.
        merged_results: Dict = {}
        
        logger.debug(f"Mix of tasks: ")
        # TODO: Write tests to check that this works.
        
        all_task_indices: Dict[Any, Tensor] = {}
        
        # Get the indices for each task.
        for task_id in unique_task_labels:
            if isinstance(task_labels, (Tensor, np.ndarray)):
                task_indices = torch.arange(batch_size)[task_labels == task_id]
            else:
                task_indices = torch.as_tensor([
                    i for i, task_label in enumerate(task_labels)
                    if task_label == task_id
                ])
            all_task_indices[task_id] = task_indices
        
        # Get the percentage of each task in the batch.
        fraction_of_batch: Dict[int, float] = {
            task_id: len(task_indices) / batch_size
            for task_id, task_indices in all_task_indices.items()
        }
        logger.debug(f"Fraction of tasks in the batch: {fraction_of_batch}")
        # Split off the input batch, do a forward pass for each sub-task.
        # (could be done in parallel but whatever.)
        # TODO: Also, not sure if this will play well with DP, DDP, etc.
        for task_id, task_indices in all_task_indices.items():
            # TODO: Maybe Use -1 instead of 'None' and always index into
            # `self.output_heads`?
            with self.temporarily_in_task(task_id):
                logger.debug(
                    f"Doing partial forward for "
                    f"{len(task_indices)/batch_size:.0%} of the batch which "
                    f"has task_id of '{task_id}'.")

                # Make a partial observation without the task labels, so that
                # super().forward will use the current output head.
                tensor_slices = {
                    name: tensor[task_indices] for name, tensor in observation.items()
                    if name != "task_labels"
                }
                partial_observation = self.Observation(**tensor_slices)
                task_forward_pass = super().forward(partial_observation)

            if not merged_results:
                # Create the merged results: a dict with empty tensors based on
                # the shape of the first results we get.
                for name, partial_result in task_forward_pass.items():
                    if isinstance(partial_result, Tensor):
                        placeholder = partial_result.new_empty([batch_size, *partial_result.shape[1:]])
                    else:
                        placeholder_tensors = {
                            name: value.new_empty([batch_size, *value.shape[1:]])
                            for name, value in partial_result.items()
                            if isinstance(value, Tensor)
                        }
                        placeholder = type(partial_result)(**placeholder_tensors)
                    merged_results[name] = placeholder

            # Merge the results from each task by setting at the right indices
            # in the placeholder tensors. 
            for name, (full_result, partial_result) in dict_intersection(merged_results, task_forward_pass):
                if isinstance(full_result, Tensor):
                    full_result[task_indices] = partial_result
                else:
                    for name, (full_tensor, partial_tensor) in dict_intersection(full_result, partial_result):
                        if full_tensor is None:
                            assert partial_tensor is None
                        else:    
                            assert isinstance(full_tensor, Tensor)
                            assert isinstance(partial_tensor, Tensor)
                            full_tensor[task_indices] = partial_tensor

        return merged_results

    @contextmanager
    def temporarily_in_task(self, task_id: Optional[int]):
        """WIP: This would be used to temporarily change the 'output_head'
        attribute,
        """
        start_task_id = self.current_task
        assert isinstance(task_id, int) or task_id is None, task_id
        self.current_task = task_id
        yield
        self.current_task = start_task_id
    
    def _shared_step(self, batch: Tuple[Tensor, Optional[Tensor]],
                           batch_idx: int,
                           dataloader_idx: int = None,
                           loss_name: str = "",
                           training: bool = True,
                    ) -> Dict:
        assert loss_name
        if dataloader_idx is not None:
            logger.debug(
                "TODO: We were indirectly given a task id with the "
                "dataloader_idx. Ignoring for now, as we're trying to avoid "
                "this (the task labels should be given for each example "
                "anyway). "
            )
            dataloader_idx = None
        elif ((training and self.setting.task_labels_at_train_time) or
              (not training and self.setting.task_labels_at_test_time)):
            # If we're not told the dataloader idx, but we have access to the
            # task labels, then switch to the current task if it's not the same
            # as the previous task.
            # TODO: Remove this, and use the per-sample task labels from
            # continuum instead.
            current_task = self.setting.current_task_id
            if self.current_task != current_task:
                self.previous_task = self.current_task
                self.current_task = current_task

                self.on_task_switch(self.current_task)
        
        return super().shared_step(
            batch=batch,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
            loss_name=loss_name,
        )

    def on_task_switch(self, task_id: Optional[int]) -> None:
        """Called when switching between tasks.
        
        Args:
            task_id (int, optional): the id of the new task. When None, we are
            basically being informed that there is a task boundary, but without
            knowing what task we're switching to.
        """
        super().on_task_switch(task_id=task_id)
        if task_id is None:
            # TODO: Try to do some kind of task inference here, if possible!
            pass    
        if task_id is not None and self.hp.multihead and str(task_id) not in self.output_heads:
            self.output_heads[str(task_id)] = self.create_output_head()

    @property
    def current_task_classes(self) -> List[int]:
        # TODO: detect wether we are training or testing.
        return self.setting.current_task_classes(self.training)

    def preprocess_batch(self, *batch) -> Tuple[Tensor, Optional[Tensor]]:
        assert False, batch
        
    #     # TODO: Sort this out a bit.
    #     x, y, *extra_inputs = super().preprocess_batch(*batch)
    #     task_labels: Optional[Tensor] = None
    #     if len(extra_inputs) >= 1:
    #         task_labels = extra_inputs[0]

    #     if task_labels is None and y is not None:
    #         # This basically checks if the labels in y have already been
    #         # relabeled in the range [0-n_classes_per_task]. If they aren't,
    #         # then raises an error.
    #         current_task_id = self.setting.current_task_id
    #         current_task_classes = self.setting.task_classes(current_task_id, self.training)
    #         logger.debug(f"Current task id: {current_task_id}")
    #         logger.debug(f"Classes in current task: {current_task_classes}")
    #         n_classes_per_task = self.setting.n_classes_per_task
    #         # y_unique are the (sorted) unique values found within the batch.
    #         # idx[i] holds the index of the value at y[i] in y_unique, s.t. for
    #         # all i in range(0, len(y)) --> y[i] == y_unique[idx[i]]
    #         y_unique, idx = y.unique(sorted=True, return_inverse=True)

            
    #         if set(y_unique.tolist()) <= set(range(n_classes_per_task)):
    #             # The images were already re-labeled by the continuum package.
    #             return x, y
    #         # TODO: This might not make sense anymore, given that I think the
    #         # Continuum package already re-labels these for us. We could however
    #         # just figure out which output head to use, if we stored the
    #         # 'classes' that were assigned to each output head during training.
    #         # (This might be similar to the "labels trick" from
    #         # https://arxiv.org/abs/1803.10123)
    #         elif not (set(y_unique.tolist()) <= set(current_task_classes)):
    #             raise RuntimeError(
    #                 f"There are labels in the batch that aren't part of the "
    #                 f"current task! (current task: "
    #                 f"{current_task_id}), Current task classes: "
    #                 f"{current_task_classes}, batch labels: {y_unique})"
    #             )
    #         else:
    #             # Relabel the images manually, which really sucks!
    #             y = self.relabel(y, current_task_classes)
    #     return x, y

    # def relabel(self, y: Tensor, current_task_classes: List[int]) -> Tensor:
    #     # Re-label the given batch so the losses/metrics work correctly.
    #     # Example: if the current task classes is [2, 3] then relabel that
    #     # those examples as [0, 1].
    #     # TODO: Double-check that that this is what is usually done in CL.
    #     new_y = torch.empty_like(y)
    #     for i, label in enumerate(current_task_classes):
    #         new_y[y == label] = i
    #     return new_y

    def load_state_dict(self, state_dict: Union[Dict[str, Tensor], Dict[str, Tensor]],
                        strict: bool = True):
        if self.hp.multihead:
            # TODO: Figure out exactly where/when/how pytorch-lightning is
            # trying to load the model from, because there are some keys
            # missing (['output_heads.1.output.weight', 'output_heads.1.output.bias'])
            # For now, we're just gonna pretend it's not a problem, I guess?
            strict = False

        missing_keys, unexpected_keys = super().load_state_dict(
            state_dict=state_dict,
            strict=False
        )

        # TODO: Double-check that this makes sense and works properly.
        if self.hp.multihead and unexpected_keys:
            for i in range(self.setting.nb_tasks):
                new_output_head = self.create_output_head()
                new_output_head.load_state_dict(
                    {k: state_dict[k] for k in unexpected_keys},
                    strict=False,
                )
                key = str(i)
                self.output_heads[key] = new_output_head.to(self.device)

        if missing_keys or unexpected_keys:
            logger.debug(f"Missing keys: {missing_keys}, unexpected keys: {unexpected_keys}")
        
        return missing_keys, unexpected_keys
