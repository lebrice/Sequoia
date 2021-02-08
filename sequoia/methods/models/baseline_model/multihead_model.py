from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union, Set, Sequence
from contextlib import contextmanager

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from pytorch_lightning.core.decorators import auto_move_data

from sequoia.common import Config, Batch, Loss

from sequoia.settings import ClassIncrementalSetting, Environment, Observations, Actions, Rewards
from sequoia.settings.assumptions.incremental import IncrementalSetting

from sequoia.utils import dict_intersection, zip_dicts, prod
from sequoia.utils.logging_utils import get_logger
from sequoia.utils.generic_functions import get_slice, set_slice
from ..forward_pass import ForwardPass
# from .semi_supervised_model import SemiSupervisedModel
from .base_model import BaseModel
from ..output_heads import OutputHead
logger = get_logger(__file__)


SettingType = TypeVar("SettingType", bound=IncrementalSetting)


class MultiHeadModel(BaseModel[SettingType]):
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
        multihead: Optional[bool] = None

    def __init__(self, setting: IncrementalSetting, hparams: HParams, config: Config):
        super().__init__(setting=setting, hparams=hparams, config=config)
        self.output_heads: Dict[str, OutputHead] = nn.ModuleDict()
        self.hp: MultiHeadModel.HParams
        self.setting: SettingType

        # TODO: Add an optional task inference mechanism for ClassIncremental
        # methods!
        self.task_inference_module: Optional[nn.Module] = None

        self.previous_task: Optional[int] = None
        self.current_task: Optional[int] = None

    @property
    def default_output_head(self) -> OutputHead:
        return self.output_heads[str(None)]

    # @property
    # def output_head(self) -> OutputHead:
    #     """ Get the output head for the current task.

    #     FIXME: It's generally bad practice to do heavy computation on a property
    #     so we should probably add something like a get_output_head(task) method.
    #     """
    #     if self.setting.nb_tasks == 1 or not self.hp.multihead:
    #         return self.output_heads[str(None)]
        
    #     # We have a multi-headed model (often means we have task labels, but not
    #     # necessarily).
    #     key = str(self.current_task)
    #     if key not in self.output_heads:
    #         self.output_heads[key] = self.create_output_head(self.setting)
    #     return self.output_heads[key]

    # @output_head.setter
    # def output_head(self, value: OutputHead) -> None:
    #     # logger.debug(f"Setting output head to {value}")
    #     # TODO: There's a problem here with multiheaded models. This setter gets
    #     # 'bypassed' somehow.
    #     assert False, value
    #     self._output_head = value

    @auto_move_data
    def forward(self, observations:  IncrementalSetting.Observations) -> Dict[str, Tensor]:
        """ Forward pass of the Model. Returns a dict."""
        # Just testing things out here.
        assert isinstance(observations, self.Observations), observations
        single_observation_space = self.observation_space
        if observations[0].shape == single_observation_space[0].shape:
            raise RuntimeError("Observations should be batched!")
        
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

        if isinstance(task_labels, (Tensor, np.ndarray)):
            unique_task_labels = list(set(task_labels.tolist()))
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

    def output_head_loss(self,
                         forward_pass: ForwardPass,
                         actions: Actions,
                         rewards: Rewards) -> Loss:
        # Asks each output head for its contribution to the loss.
        # TODO: Get the loss of the output heads.

        observations: IncrementalSetting.Observations = forward_pass.observations
        task_labels = observations.task_labels
        batch_size = forward_pass.batch_size
        assert batch_size is not None

        if not self.hp.multihead:
            # Default behaviour: use the (only) output head.
            return super().output_head_loss(forward_pass, actions=actions, rewards=rewards)

        if task_labels is None:
            if self.task_inference_module:
                # TODO: Predict the task ids using some kind of task
                # inference mechanism.
                task_labels = self.task_inference_module(forward_pass)
            else:
                raise NotImplementedError(
                    f"Multihead model doesn't have access to task labels and "
                    f"doesn't have a task inference module!"
                )
                # TODO: Maybe use the last trained output head, by default.

        assert task_labels is not None
        unique_task_labels: List[Optional[int]] = list(set(task_labels.tolist()))

        if len(unique_task_labels) == 1:
            # If everything is in the same task, no need to split/merge.
            task_id = unique_task_labels[0]
            if self.current_task == task_id:
                # Already in that task:
                loss = super().output_head_loss(forward_pass, actions=actions, rewards=rewards)
                # FIXME: Debugging stuff.
                # loss.name += f"(task {self.current_task})"
                return loss

            # TODO: This is messing things up in RL!
            if self.training and self.hp.batch_size == 1:
                # IDEA: Maybe in this case we can safely destroy any preserved state in
                # the current output head.
                assert isinstance(task_id, int), "(wip) assuming we have task ids here."
                # TODO: this isn't really pretty, but the idea is that we need to
                # actually flush out any 
                logger.debug(f"Manually calling on_task_switch({task_id}) since we're "
                             f"training in a single env, which is most probably a "
                             f"multi-task RL environment.")
                self.on_task_switch(task_id)
                loss = super().output_head_loss(forward_pass, actions=actions, rewards=rewards)
                return loss

            # Switch tasks "temporarily".
            # TODO: This can make things quite complicated in RL, as the output heads
            # currently have state for the environments they are being trained on.             
            with self.temporarily_in_task(task_id):
                # Only one loss to fetch, since all items are from the same task.
                # Default behaviour: use the (only) output head.
                loss = super().output_head_loss(forward_pass, actions=actions, rewards=rewards)
                # loss.name += f"(task {self.current_task})"
                return loss

        all_task_indices: Dict[Any, Tensor] = {}
        
        # Get the indices for each task.
        for task_id in unique_task_labels:
            if isinstance(task_labels, np.ndarray):
                task_indices = np.arange(batch_size)[task_labels == task_id]
            if isinstance(task_labels, Tensor):
                task_indices = torch.arange(batch_size)[task_labels == task_id]
            else:
                task_indices = torch.as_tensor([
                    i for i, task_label in enumerate(task_labels)
                    if task_label == task_id
                ])
            all_task_indices[task_id] = task_indices

        total_loss = Loss(self.output_head.name)

        # Split off the input batch, do a forward pass for each sub-task.
        # (could be done in parallel but whatever.)
        # TODO: Also, not sure if this will play well with DP, DDP, etc.
        for task_id, task_indices in all_task_indices.items():
            # # Make a partial observation without the task labels, so that
            # # super().forward will use the current output head.
            forward_pass_slice = get_slice(forward_pass, task_indices)
            actions_slice = get_slice(actions, task_indices)
            rewards_slice = get_slice(rewards, task_indices)

            logger.debug(
                f"Getting output head loss"
                f"{len(task_indices)/batch_size:.0%} of the batch which "
                f"has task_id of '{task_id}'."
            )
            
            with self.temporarily_in_task(task_id):
                task_output_head_loss = super().output_head_loss(
                    forward_pass_slice,
                    actions=actions_slice,
                    rewards=rewards_slice,
                )
                # FIXME: debugging
                # task_output_head_loss.name += f"(task {task_id})"
                logger.debug(f"Task {task_id} loss: {task_output_head_loss}")
                total_loss += task_output_head_loss

        return total_loss


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
        if task_id != self.current_task:
            logger.debug(f"Destroying all buffer contents in the output heads.")
            logger.debug(f"self.current_task = {self.current_task}, new task: {task_id})")
            self.output_head.clear_all_buffers()
            for output_head in self.output_heads.values():
                output_head.clear_all_buffers()

        super().on_task_switch(task_id=task_id)
        logger.info(f"Switching from task {self.current_task} -> {task_id}.")
        self.previous_task = self.current_task
        self.current_task = task_id

        if task_id is None:
            # TODO: Try to do some kind of task inference here, if possible!
            # TODO: Should we revert back to using a 'default' output head?
            # ('None' key?) or just use the last trained output head?
            # self.output_head = self.output_heads[str(None)]
            pass
                        
        # TODO: Do we need to 'save' the output head back into
        # `self.output_heads`? do `self.output_head` and
        # `self.output_heads[str(self.previous_task)]` reference the same
        # object? or does assigning a new value to self.output_head perform a
        # copy under the hood in nn.Module?
        if str(self.previous_task) in self.output_heads:
            assert id(self.output_head) == id(self.output_heads[str(self.previous_task)])
        self.output_heads[str(self.previous_task)] = self.output_head

        key = str(task_id)
        if self.hp.multihead:
            if key not in self.output_heads:
                logger.info(f"Creating a new output head for task {key}.")
                self.output_heads[key] = self.create_output_head(self.setting, task_id=task_id)
            # Update `self.output_head` to be the one for the current task.
            self.output_head = self.output_heads[key]

        # NOTE: IF the model *isn't* multi-headed, then we always use the output
        # head at key 'None' anyway, so we don't create a new head here.

        
    @contextmanager
    def temporarily_in_task(self, task_id: Optional[int]):
        """WIP: This would be used to temporarily change the 'output_head'
        attribute,
        """
        start_task_id = self.current_task
        start_output_head = self.output_head
        assert isinstance(task_id, int) or task_id is None

        output_head_key = str(task_id)
        if self.hp.multihead and task_id is None:
            # Multi-headed model, but we don't know the task id: need to use
            # some kind of task inference module?
            raise NotImplementedError("todo")
        elif not self.hp.multihead:
            # We are using a single-head model, so we will use the 'default'
            # output head. 
            output_head_key = str(None)

        self.current_task = task_id
        # NOTE: May need to create new output heads here, since on_task_switch isn't
        # always called before we see data of a new task (as is the case in so-called
        # "Multi-Task" RL.)
        if output_head_key not in self.output_heads:
            logger.info(f"Creating a new output head for task {output_head_key}.")
            new_output_head = self.create_output_head(self.setting, task_id=task_id)
            self.output_heads[output_head_key] = new_output_head

        # TODO: BUG: There is "old" state left in the buffers of the output head from
        # previous forward/backward passes!
        # Need to clear the output head's state somehow when we're done with it, but
        # also somehow allow it to accumulate state when it is being applied on the same
        # task over multiple steps!
        if self.output_head is not self.output_heads[output_head_key]:
            from sequoia.methods.models.output_heads.rl import PolicyHead
            if isinstance(self.output_head, PolicyHead):
                self.output_head.clear_all_buffers()

        self.output_head = self.output_heads[output_head_key]

        yield
        # TODO: Not sure we need to do this, but just to be safe:
        self.output_heads[output_head_key] = self.output_head

        # Restore the previous task id and output head.
        self.current_task = start_task_id
        self.output_head = start_output_head

    @property
    def current_task_classes(self) -> List[int]:
        # TODO: detect wether we are training or testing.
        return self.setting.current_task_classes(self.training)
   
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
                logger.info(f"Creating a new output head for task {i}")
                new_output_head = self.create_output_head(self.setting, task_id=i)
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
    """ IDEA: Creates a 'placeholder', which will be later populated with the values
    from different tasks.
    """
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

from sequoia.utils.categorical import Categorical

@create_placeholder.register(Categorical)
def _create_placeholder_categorical(original: Categorical, batch_size: int) -> Tuple[T]:
    placeholder = type(original)(
        logits=torch.randn(
            [batch_size, *original.logits.shape[1:]],
            dtype=original.logits.dtype,
            device=original.logits.device,
        )
    )
    return placeholder

Dataclass = TypeVar("Dataclass", bound=Batch)


# @create_placeholder.register(NamedTuple)
@create_placeholder.register(Batch)
def _create_placeholder_dataclass(original: Dataclass, batch_size: int) -> Dataclass:
    return type(original)(**{
        key: create_placeholder(value, batch_size)
        for key, value in original.items()    
    })
