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


        # TODO: Add an optional task inference mechanism
        # See https://github.com/lebrice/Sequoia/issues/49 
        self.task_inference_module: Optional[nn.Module] = None

        self.previous_task: Optional[int] = None
        self.current_task: Optional[int] = None

        self.previous_task_labels: Optional[Sequence[int]] = None

    @property
    def default_output_head(self) -> OutputHead:
        return self.output_heads["0"]

    @contextmanager
    def switch_output_head(self, task_id: int):
        """Temporarily switches out the output head for the one for task `task_id`.
        
        Also temporarily changes the value of `self.current_task`.
        If `task_id` is not a known task and doesn't already have an associated output
        head, then a new output head is created and stored in the `output_heads` dict.

        TODO: Not sure if there would be some value in making this a bit more 'general',
        since after all the entire forward pass is "multiplexed"
        
        Parameters
        ----------
        task_id : int
            The index of the task to switch to.
        """
        assert isinstance(task_id, int), f"Not sure what to do! (task_id={task_id})"
        starting_output_head = self.output_head
        starting_task = self.current_task

        # Only perform this 'switch' if need to.
        if task_id != self.current_task:
            # Note: ModuleDicts only accept string keys, for some reason.
            if str(task_id) not in self.output_heads:
                task_output_head = self.create_output_head(self.setting, task_id=task_id)
                self.output_heads[str(task_id)] = task_output_head
            else:
                task_output_head = self.output_heads[str(task_id)]

            self.current_task = task_id
            self.output_head = task_output_head

            logger.debug(f"Switching output heads")
        # Yield to "give back control" to the inner portion of the 'with' statement.
        yield

        # Reset the original values.
        self.current_task = starting_task
        self.output_head = starting_output_head

    @auto_move_data
    def forward(self, observations: IncrementalSetting.Observations) -> ForwardPass:
        """Forward pass of the Model. Performs a split-batch forward for each task.
        
        IDEA: This calls super.forward() on the slices of the batch for each task, and
        then re-combines the forward passes from each task into a single result.
        It's a bit extra. Maybe we only really ever want to have the output task be the
        'branched-out/multi-task' portion.

        Parameters
        ----------
        observations : IncrementalSetting.Observations
            Observations from an environment. So far, this will always be from an
            `IncrementalSetting`, i.e. descendant of `ContinualRLSetting` or
            `ClassIncrementalSetting`.

        Returns
        -------
        ForwardPass
            A merged ForwardPass object containing the forward pass for each task.
        """
        # The forward pass to be returned:
        forward_pass: Optional[ForwardPass] = None

        if not self.batch_size:
            self.batch_size = observations.batch_size
            logger.debug(f"Setting batch_size to {self.batch_size}.")

        # Just testing things out here.
        assert isinstance(observations, self.Observations), observations
        single_observation_space = self.observation_space
        if observations[0].shape == single_observation_space[0].shape:
            raise RuntimeError("Observations should be batched!")

        # Get the task labels from the observation.
        # TODO: It isn't exactly nice that we have to do this here. Would be nicer if we
        # always had task labels for each sample as a numpy array, or just None.
        task_labels: Optional[np.ndarray] = cleanup_task_labels(observations.task_labels)
        # Get the indices corresponding to the elements from each task within the batch.
        task_indices: Dict[Optional[int], np.ndarray] = get_task_indices(task_labels)

        if task_labels is None:
            # Default back to the behaviour of the base class, which will use
            # the current output head (at attribute `self.output_head`), whatever that
            # may be.
            forward_pass = super().forward(observations)

        elif len(task_indices) == 1:
            # If everything is in the same task, no need to split/merge stuff, which is
            # a bit easier to deal with.
            task_id = list(task_indices.keys())[0]

            if task_id != self.current_task:
                logger.warning(
                    RuntimeWarning(
                        f"All data in the batch comes from task {task_id}, but the "
                        f"current task is set to {self.current_task}.. "
                        f"Calling on_task_switch({task_id}) manually?."
                    )
                )
                # TODO: Not sure about this!
                self.on_task_switch(task_id)
            forward_pass = super().forward(observations)

        else:
            logger.debug(f"Batch contains a mix of tasks!")
            batch_size = len(task_labels)
            # Split off the input batch, do a forward pass for each sub-task.
            # (could be done in parallel but whatever.)
            for task_id, task_indices in task_indices.items():
                # Take the elements for that task and create a new Observation of the
                # same type.
                partial_observation = get_slice(observations, task_indices)
                logger.debug(
                    f"Doing partial forward for "
                    f"{len(task_indices)/batch_size:.0%} of the batch which "
                    f"has task_id of '{task_id}'."
                )

                # TODO: Here instead of calling on_task_switch, or anything fancy, I think
                # it might be simplest to just change the output head for now.
                with self.switch_output_head(task_id):
                    task_forward_pass = super().forward(partial_observation)

                if not forward_pass:
                    # Create the merged results, filled with empty tensors, based on
                    # the shape of the first results we get, but with the right
                    # batch size.
                    forward_pass = create_placeholder(task_forward_pass, batch_size)

                # Set the partial results at the right indices in the placeholders.
                set_slice(forward_pass, task_indices, task_forward_pass)

        assert forward_pass
        return forward_pass

    def output_head_loss(
        self, forward_pass: ForwardPass, actions: Actions, rewards: Rewards
    ) -> Loss:
        # Asks each output head for its contribution to the loss.
        observations: IncrementalSetting.Observations = forward_pass.observations
        task_labels = observations.task_labels
        batch_size = forward_pass.batch_size
        assert batch_size is not None
        
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
                # TODO: Maybe use the last trained output head, by default?
        # BUG: We get no loss from the output head for the first episode after a task
        # switch.
        # NOTE: The problem is that the `done` in the observation isn't necessarily
        # associated with the task designed by the `task_id` in that observation!
        # That is because of how vectorized environments work, they reset the env and
        # give the new initial observation when `done` is True, rather than the last
        # observation in that env.
        if self.previous_task_labels is None:
            self.previous_task_labels = task_labels

        # Default behaviour: use the (only) output head.
        if not self.hp.multihead:
            return self.output_head.get_loss(
                forward_pass, actions=actions, rewards=rewards,
            )

        # The sum of all the losses from all the output heads.    
        total_loss = Loss(self.output_head.name)
                
                
        task_switched_in_env = (task_labels != self.previous_task_labels)
        episode_ended = observations.done
        # logger.debug(f"Task labels: {task_labels}, task switched in env: {task_switched_in_env}, episode ended: {episode_ended}")
        done_set_to_false_temporarily_indices = []

        if any(episode_ended & task_switched_in_env):
            # In the environments where there was a task switch to a different task and
            # where some episodes ended, we need to first get the corresponding output
            # head losses from these environments first.
            if self.batch_size in {None, 1}:
                # If the batch size is 1, this is a little bit simpler to deal with.
                previous_task: int = self.previous_task_labels[0].item()
                # IDEA:
                from sequoia.methods.models.output_heads.rl import PolicyHead
                previous_output_head = self.output_heads[str(previous_task)]
                assert isinstance(previous_output_head, PolicyHead), "todo: assuming that this only happends in RL currently."
                # We want the loss from that output head, but we don't want to
                # re-compute it below!
                env_index_in_previous_batch = 0
                # breakpoint()
                logger.debug(f"Getting a loss from the output head for task {previous_task}, that was used for the last task.")
                env_episode_loss = previous_output_head.get_episode_loss(env_index_in_previous_batch, done=True)
                # logger.debug(f"Loss from that output head: {env_episode_loss}")
                # Add this end-of-episode loss to the total loss.
                # breakpoint()
                assert env_episode_loss is not None
                total_loss += env_episode_loss
                previous_output_head.on_episode_end(env_index_in_previous_batch)

                # Set `done` to `False` for that env, to prevent the output head for the
                # new task from seeing the first observation in the episode as the last.
                observations.done[env_index_in_previous_batch] = False
                done_set_to_false_temporarily_indices.append(env_index_in_previous_batch)
                # BUG: If we modify that entry in-place, then even after the end of this
                # method the change persists..
            else:
                raise NotImplementedError(f"TODO: Need to somehow pass the indices of "
                                          f"which env to take care of to each output "
                                          f"head, so they can create / clear buffers "
                                          f"only when needed.")

        assert task_labels is not None
        all_task_indices: Dict[int, Tensor] = get_task_indices(task_labels)

        # Get the loss from each output head:
        if len(all_task_indices) == 1:
            # If everything is in the same task (only one key), no need to split/merge
            # stuff, so it's a bit easier:
            task_id: int = list(all_task_indices.keys())[0]
            
            with self.switch_output_head(task_id):
                # task_output_head = self.output_heads[str(task_id)]
                total_loss += self.output_head.get_loss(
                    forward_pass, actions=actions, rewards=rewards,
                )
        else:
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
                task_output_head = self.output_heads[str(task_id)]
                task_loss = task_output_head.get_loss(
                    forward_pass_slice, actions=actions_slice, rewards=rewards_slice,
                )
                # FIXME: debugging
                # task_output_head_loss.name += f"(task {task_id})"
                logger.debug(f"Task {task_id} loss: {task_loss}")
                total_loss += task_loss

        self.previous_task_labels = task_labels
        # FIXME: Reset the 'done' to True, if we manually set it to False.
        for index in done_set_to_false_temporarily_indices:
            observations.done[index] = True
        
        return total_loss

    def on_after_backward(self):
        super().on_after_backward()

    def on_before_zero_grad(self, optimizer):
        super().on_before_zero_grad(optimizer)
        from sequoia.methods.models.output_heads.rl import PolicyHead
        for task_id_string, output_head in self.output_heads.items():
            if isinstance(output_head, PolicyHead):
                output_head: PolicyHead
                output_head.detach_all_buffers()

        
    def shared_step(
        self,
        batch: Tuple[Observations, Optional[Rewards]],
        batch_idx: int,
        environment: Environment,
        loss_name: str,
        dataloader_idx: int = None,
        optimizer_idx: int = None,
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

        return super().shared_step(
            batch=batch,
            batch_idx=batch_idx,
            environment=environment,
            loss_name=loss_name,
            dataloader_idx=dataloader_idx,
            optimizer_idx=optimizer_idx,
        )

    def on_task_switch(
        self, task_id: Optional[int], clear_buffers: bool = False
    ) -> None:
        """Called when switching between tasks.
        
        Args:
            task_id (int, optional): the id of the new task. When None, we are
            basically being informed that there is a task boundary, but without
            knowing what task we're switching to.
        """
        # if task_id != self.current_task:
        #     logger.debug(f"Destroying all buffer contents in the output heads.")
        #     logger.debug(f"self.current_task = {self.current_task}, new task: {task_id})")
        #     self.output_head.clear_all_buffers()
        #     for output_head in self.output_heads.values():
        #         output_head.clear_all_buffers()

        logger.info(f"Switching from task {self.current_task} -> {task_id}.")

        super().on_task_switch(task_id=task_id)

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
            assert id(self.output_head) == id(
                self.output_heads[str(self.previous_task)]
            )
        self.output_heads[str(self.previous_task)] = self.output_head

        key = str(task_id)
        if self.hp.multihead:
            if key not in self.output_heads:
                logger.info(f"Creating a new output head for task {key}.")
                self.output_heads[key] = self.create_output_head(
                    self.setting, task_id=task_id
                )
            # Update `self.output_head` to be the one for the current task.
            self.output_head = self.output_heads[key]

        # NOTE: IF the model *isn't* multi-headed, then we always use the output
        # head at key 'None' anyway, so we don't create a new head here.

    @contextmanager
    def temporarily_in_task(self, task_id: Optional[int]):
        """ This is used to temporarily change the 'output_head' attribute.
        """
        logger.debug(f"Temporarily switching to task {task_id}")
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

        # # TODO: BUG: There is "old" state left in the buffers of the output head from
        # # previous forward/backward passes!
        # # Need to clear the output head's state somehow when we're done with it, but
        # # also somehow allow it to accumulate state when it is being applied on the same
        # # task over multiple steps!

        # TODO: IDEA: Rather than try to clear this state ourselves here or in
        # `on_task_switch`, we could add some sort of method on the OutputHead class
        # that gets called before/after the model update, so that we get to detach all
        # the tensors and clear any buffers that need to be cleared, once the model has
        # performed an update.

        # TODO: The RL output heads and interleaved episodes in different tasks will
        # most definitely not work with our current mechanism for the multi-headed
        # model. Would need to share the buffers between the output heads, and then
        # indicate to each head the indices of the environments it is responsible for
        # somehow..
        self.output_head = self.output_heads[output_head_key]

        # Yield, during which the forward pass or whatever else will be performed.
        yield

        # Reset everything to their starting values.

        # TODO: Not sure we need to do this, but just to be safe:
        self.output_heads[output_head_key] = self.output_head

        # Restore the previous task id and output head.
        self.current_task = start_task_id
        self.output_head = start_output_head

    @property
    def current_task_classes(self) -> List[int]:
        # TODO: detect wether we are training or testing.
        return self.setting.current_task_classes(self.training)

    def load_state_dict(
        self,
        state_dict: Union[Dict[str, Tensor], Dict[str, Tensor]],
        strict: bool = True,
    ):
        if self.hp.multihead:
            # TODO: Figure out exactly where/when/how pytorch-lightning is
            # trying to load the model from, because there are some keys
            # missing (['output_heads.1.output.weight', 'output_heads.1.output.bias'])
            # For now, we're just gonna pretend it's not a problem, I guess?
            strict = False

        missing_keys, unexpected_keys = super().load_state_dict(
            state_dict=state_dict, strict=False
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
                    {k: state_dict[k] for k in unexpected_keys}, strict=False,
                )
                key = str(i)
                self.output_heads[key] = new_output_head.to(self.device)

        if missing_keys or unexpected_keys:
            logger.debug(
                f"Missing keys: {missing_keys}, unexpected keys: {unexpected_keys}"
            )

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
        (key, create_placeholder(value, batch_size)) for key, value in original.items()
    )


@create_placeholder.register(tuple)
def _create_placeholder_tuple(original: Tuple[T], batch_size: int) -> Tuple[T]:
    return type(original)(create_placeholder(value, batch_size) for value in original)


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

# IDEA: Maybe replace `create_placeholder` with simply `Batch.new_empty()` or something
# similar?

# @create_placeholder.register(NamedTuple)
@create_placeholder.register(Batch)
def _create_placeholder_dataclass(original: Dataclass, batch_size: int) -> Dataclass:
    return type(original)(
        **{
            key: create_placeholder(value, batch_size)
            for key, value in original.items()
        }
    )

def get_task_indices(task_labels: Union[List[Optional[int]], np.ndarray, Tensor]) -> Dict[Optional[int], Union[np.ndarray, Tensor]]:
    """Given an array-like of task labels, gives back a dictionary mapping from task id
    to an array-like of indices for the corresponding indices in the batch. 

    Parameters
    ----------
    task_labels : Union[np.ndarray, Tensor]
        [description]

    Returns
    -------
    Dict[Optional[int], Union[np.ndarray, Tensor]]
        Dictionary mapping from task index (int or None) to an ndarray or Tensor
        (depending on the type of `task_labels`) of indices corresponding to the indices
        in `task_labels` that correspond to that task.
    """
    all_task_indices: Dict[int, Union[np.ndarray, Tensor]] = {}
    
    if task_labels is None:
        return {}
    
    if isinstance(task_labels, (Tensor, np.ndarray)):
        task_labels = task_labels.tolist()
    else:
        # In case task_labels is a list of numpy arrays, convert it to a
        # list of elements (optional ints).
        task_labels = [
            int(label) if label != None else None for label in task_labels
        ]
    unique_task_labels = list(set(task_labels))
    
    batch_size = len(task_labels)
    # Get the indices for each task.
    for task_id in unique_task_labels:
        if isinstance(task_labels, np.ndarray):
            task_indices = np.arange(batch_size)[task_labels == task_id]
        if isinstance(task_labels, Tensor):
            task_indices = torch.arange(batch_size)[task_labels == task_id]
        else:
            task_indices = torch.as_tensor(
                [
                    i
                    for i, task_label in enumerate(task_labels)
                    if task_label == task_id
                ]
            )
        all_task_indices[task_id] = task_indices 
    return all_task_indices


def cleanup_task_labels(task_labels: Optional[Sequence[Optional[int]]]) -> Optional[np.ndarray]:
    """ 'cleans up' the task labels, by returning either None or an integer numpy array.

    TODO: Not clear why we really have to do this in the first place. The point is, if
    we wanted to allow only a fraction of task labels for instance, then we have to deal
    with np.ndarrays with `object` dtypes.
    
    Parameters
    ----------
    task_labels : Optional[Sequence[Optional[int]]]
        Some sort of array of task ids, or None.

    Returns
    -------
    Optional[np.ndarray]
        None if there are no task ids, or an integer numpy array if there are.

    Raises
    ------
    NotImplementedError
        If only a portion of the task labels are available.
    """
    if isinstance(task_labels, np.ndarray):
        if task_labels.dtype == object:
            if all(task_labels == None):
                task_labels = None
            elif all(task_labels != None):
                task_labels = torch.as_tensor(task_labels.astype(np.int))
            else:
                raise NotImplementedError(
                    f"TODO: Only given a portion of task labels?"
                )
                # IDEA: Maybe set task_id to -1 in those cases, and return an int
                # ndarray as well?
    if not task_labels.shape:
        task_labels = task_labels.reshape([1])
    if isinstance(task_labels, Tensor):
        task_labels = task_labels.cpu().numpy()
    if task_labels is not None:
        task_labels = task_labels.astype(int)
    assert task_labels is None or isinstance(task_labels, np.ndarray), (task_labels, task_labels.dtype)
    return task_labels