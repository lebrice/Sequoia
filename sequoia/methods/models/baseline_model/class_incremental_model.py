from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union, Set, Sequence
from contextlib import contextmanager

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from pytorch_lightning.core.decorators import auto_move_data

from sequoia.common.config import Config
from sequoia.common.batch import Batch

from sequoia.settings import ClassIncrementalSetting, Environment, Observations, Actions, Rewards
from sequoia.settings.assumptions.incremental import IncrementalSetting

from sequoia.utils import dict_intersection, zip_dicts, prod
from sequoia.utils.logging_utils import get_logger
from sequoia.utils.generic_functions import get_slice, set_slice

# from .semi_supervised_model import SemiSupervisedModel
from .base_model import BaseModel
from ..output_heads import OutputHead
logger = get_logger(__file__)


SettingType = TypeVar("SettingType", bound=IncrementalSetting)


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

    def __init__(self, setting: IncrementalSetting, hparams: HParams, config: Config):
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
            output_head = self.create_output_head(self.setting)
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
                output_head = self.create_output_head(self.setting)
                self.output_heads[key] = output_head
            else:
                output_head = self.output_heads[key]
            self._output_head = output_head
            # Return the output head for the current task.
            return output_head

        if self._output_head is None:
            self._output_head = self.create_output_head(self.setting)
        return self._output_head

    @output_head.setter
    def output_head(self, value: OutputHead) -> None:
        # logger.debug(f"Setting output head to {value}")
        self._output_head = value

    @auto_move_data
    def forward(self, observations:  IncrementalSetting.Observations) -> Dict[str, Tensor]:
        """ Forward pass of the Model. Returns a dict."""
        # Just testing things out here.
        assert isinstance(observations, self.Observations), observations
        single_observation_space = self.observation_space
        if observations in single_observation_space:
            raise RuntimeError(
                f"Observations should be batched! (shapes: {observations.shapes}, space: {single_observation_space})"
            )
        # Get the task labels from the observation.
        task_labels = observations.task_labels
        
        if isinstance(task_labels, np.ndarray):
            if task_labels.dtype == np.object:
                if all(task_labels == None):
                    task_labels = None
                elif all(task_labels != None):
                    task_labels = torch.as_tensor(task_labels.as_dtype(np.int))
                else:
                    raise NotImplementedError(f"TODO: Only given a portion of task labels?")
        if isinstance(task_labels, Tensor) and not task_labels.shape:
            task_labels = task_labels.reshape([1])
  
        # IDEA: This would basically call super.forward() on the slices of the
        # batch, and then re-combine the forward pass dicts before returning
        # the results.
        # It's a bit extra. Maybe we only really ever want to have the output
        # task be the 'branched-out/multi-task' portion.
        if task_labels is None:
            # Default back to the behaviour of the parent class, which will use
            # the current output head (at attribute `self.output_head`).
            return super().forward(observations)

        if isinstance(task_labels, Tensor):
            unique_task_labels = torch.unique(task_labels).tolist()
        else:

            # In case task_labels is a list of numpy arrays, convert it to a
            # list of elements (optional ints).
            task_labels = [int(label) if label != None else None for label in task_labels]
            unique_task_labels = list(set(task_labels))

        if len(unique_task_labels) == 1:
            # If everything is in the same task, no need to split/merge.
            task_id = unique_task_labels[0]
            with self.temporarily_in_task(task_id):
                return super().forward(observations)

        batch_size = observations.batch_size

        # The 'merged' forward pass result dict.
        merged_forward_pass: Dict = {}
        
        logger.debug(f"Batch contains a mix of tasks!")
        
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
            # # Make a partial observation without the task labels, so that
            # # super().forward will use the current output head.
            partial_observation = get_slice(observations, task_indices)

            logger.debug(
                f"Doing partial forward for "
                f"{len(task_indices)/batch_size:.0%} of the batch which "
                f"has task_id of '{task_id}'."
            )
            
            with self.temporarily_in_task(task_id):
                task_forward_pass = super().forward(partial_observation)
                # print(f"forward pass of task {task_id}: {task_forward_pass}")

            if not merged_forward_pass:
                # Create the merged results, filled with empty tensors, based on
                # the shape of the first results we get, but with the right
                # batch size.
                merged_forward_pass = create_placeholder(task_forward_pass, batch_size)

            # Set the partial results at the right indices in the placeholders. 
            set_slice(merged_forward_pass, task_indices, task_forward_pass) 
           
        return merged_forward_pass

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
    
    def shared_step(self,
                    batch: Tuple[Observations, Optional[Rewards]],
                    batch_idx: int,
                    environment: Environment,
                    loss_name: str,
                    dataloader_idx: int = None,
                    optimizer_idx: int = None) -> Dict:
        assert loss_name
        if dataloader_idx is not None:
            logger.debug(
                "TODO: We were indirectly given a task id with the "
                "dataloader_idx. Ignoring for now, as we're trying to avoid "
                "this (the task labels should be given for each example "
                "anyway). "
            )
            dataloader_idx = None
        
        return super().shared_step(
            batch=batch,
            batch_idx=batch_idx,
            environment=environment,
            loss_name=loss_name,
            dataloader_idx=dataloader_idx,
            optimizer_idx=optimizer_idx,
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
            self.output_heads[str(task_id)] = self.create_output_head(self.setting)

    @property
    def current_task_classes(self) -> List[int]:
        # TODO: detect wether we are training or testing.
        return self.setting.current_task_classes(self.training)

    def preprocess_batch(self, *batch) -> Tuple[Tensor, Optional[Tensor]]:
        # TODO: Clean this up.
        assert False, batch
   
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
                # Try to load the output head weights
                new_output_head = self.create_output_head(self.setting)
                # FIXME: TODO: This is wrong. We should create all the
                # output heads if they aren't already created, and then try to
                # load the state_dict again.
                new_output_head.load_state_dict(
                    {k: state_dict[k] for k in unexpected_keys},
                    strict=False,
                )
                key = str(i)
                self.output_heads[key] = new_output_head.to(self.device)

        if missing_keys or unexpected_keys:
            logger.debug(f"Missing keys: {missing_keys}, unexpected keys: {unexpected_keys}")
        
        return missing_keys, unexpected_keys


from functools import singledispatch
from typing import Any, Tuple, Dict, TypeVar
from sequoia.utils import NamedTuple
K = TypeVar("K")
V = TypeVar("V")
T = TypeVar("T")


@singledispatch
def create_placeholder(original: Any, batch_size: int) -> Any:
    raise NotImplementedError(original)


@create_placeholder.register(Tensor)
def _create_placeholder_tensor(original: Tensor, batch_size: int) -> Tensor:
    return original.new_empty([batch_size, *original.shape[1:]])


@create_placeholder.register(dict)
def _create_placeholder_dict(original: Dict[K, V], batch_size: int) -> Dict[K, V]:
    return type(original)(
        (key, create_placeholder(value, batch_size))
        for key, value in original.items()    
    )


@create_placeholder.register(tuple)
def _create_placeholder_tuple(original: Tuple[T], batch_size: int) -> Tuple[T]:
    return type(original)(
        create_placeholder(value, batch_size)
        for value in original
    )

Dataclass = TypeVar("Dataclass", bound=Batch)


# @create_placeholder.register(NamedTuple)
@create_placeholder.register(Batch)
def _create_placeholder_dataclass(original: Dataclass, batch_size: int) -> Dataclass:
    return type(original)(**{
        key: create_placeholder(value, batch_size)
        for key, value in original.items()    
    })
