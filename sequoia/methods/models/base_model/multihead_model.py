from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sequoia.common import Batch, Config, Loss
from sequoia.settings import Actions, Environment, Observations, Rewards
from sequoia.settings.assumptions.incremental import IncrementalAssumption
from sequoia.utils.generic_functions import concatenate, get_slice, stack
from sequoia.utils.logging_utils import get_logger

from ..forward_pass import ForwardPass
from ..output_heads import OutputHead
from .model import Model, SettingType

logger = get_logger(__file__)


class MultiHeadModel(Model[SettingType]):
    """Mixin that adds multi-head prediction to the Model when task labels are
    available.
    """

    @dataclass
    class HParams(Model.HParams):
        """Hyperparameters specific to a multi-head model."""

        # Wether to create one output head per task.
        multihead: Optional[bool] = None

    def __init__(self, setting: SettingType, hparams: HParams, config: Config):
        super().__init__(setting=setting, hparams=hparams, config=config)

        # Dictionary of output heads!
        self.output_heads: Dict[str, OutputHead] = nn.ModuleDict()
        self.hp: MultiHeadModel.HParams
        self.setting: SettingType

        # TODO: Add an optional task inference mechanism
        # See https://github.com/lebrice/Sequoia/issues/49
        self.task_inference_module: Optional[nn.Module] = None

        self.previous_task: Optional[int] = None
        self.current_task: Optional[int] = None

        self.previous_task_labels: Optional[Sequence[int]] = None

        if setting.task_labels_at_train_time:
            # NOTE: Not sure if this could cause an issue when setting is a SettingProxy
            starting_task_id = 0  # setting.current_task_id
        else:
            starting_task_id = None
        self.output_heads[str(starting_task_id)] = self.output_head

    def output_head_loss(
        self, forward_pass: ForwardPass, actions: Actions, rewards: Rewards
    ) -> Loss:
        """TODO: Need to then re-split stuff (undo the work we did in forward) to get a
        loss per output head?
        """
        # Asks each output head for its contribution to the loss.
        observations: IncrementalAssumption.Observations = forward_pass.observations
        task_labels = observations.task_labels
        if isinstance(task_labels, Tensor):
            task_labels = task_labels.cpu().numpy()

        batch_size = forward_pass.batch_size
        assert batch_size is not None

        if task_labels is None:
            if self.task_inference_module:
                # TODO: Predict the task ids using some kind of task
                # inference mechanism.
                task_labels = self.task_inference_module(forward_pass)
            else:
                raise NotImplementedError(
                    "Multihead model doesn't have access to task labels and "
                    "doesn't have a task inference module!"
                )
                # TODO: Maybe use the last trained output head, by default?

        # TODO: Check if this is still necessary
        if self.previous_task_labels is None:
            self.previous_task_labels = task_labels

        # Default behaviour: use the (only) output head.
        if not self.hp.multihead:
            return self.output_head.get_loss(
                forward_pass,
                actions=actions,
                rewards=rewards,
            )

        # The sum of all the losses from all the output heads.
        total_loss = Loss(self.output_head.name)

        task_switched_in_env = task_labels != self.previous_task_labels
        # This `done` attribute isn't added in supervised settings.
        episode_ended = getattr(observations, "done", np.zeros(batch_size, dtype=bool))
        # TODO: Remove all this useless conversion from Tensors to ndarrays
        if isinstance(episode_ended, Tensor):
            episode_ended = episode_ended.cpu().numpy()

        # logger.debug(f"Task labels: {task_labels}, task switched in env: {task_switched_in_env}, episode ended: {episode_ended}")
        done_set_to_false_temporarily_indices = []

        if any(episode_ended & task_switched_in_env):
            # In the environments where there was a task switch to a different task and
            # where some episodes ended, we need to first get the corresponding output
            # head losses from these environments first.
            if self.batch_size in {None, 1}:
                # If the batch size is 1, this is a little bit simpler to deal with.
                previous_task: int = self.previous_task_labels[0].item()
                from sequoia.methods.models.output_heads.rl import PolicyHead

                previous_output_head = self.output_heads[str(previous_task)]
                assert isinstance(
                    previous_output_head, PolicyHead
                ), "todo: assuming that this only happends in RL currently."
                # We want the loss from that output head, but we don't want to
                # re-compute it below!
                env_index_in_previous_batch = 0
                # breakpoint()
                logger.debug(
                    f"Getting a loss from the output head for task {previous_task}, that was used for the last task."
                )
                env_episode_loss = previous_output_head.get_episode_loss(
                    env_index_in_previous_batch, done=True
                )
                # logger.debug(f"Loss from that output head: {env_episode_loss}")
                # Add this end-of-episode loss to the total loss.
                # breakpoint()
                # BUG: This can sometimes (rarely) be None! Need to better understand
                # why this is happening.
                if env_episode_loss is None:
                    logger.warning(
                        RuntimeWarning(
                            f"BUG: Env {env_index_in_previous_batch} gave back a loss "
                            f"of `None`, when we expected a loss from that output head "
                            f"for task id {previous_task}."
                        )
                    )
                else:
                    total_loss += env_episode_loss
                # We call on_episode_end so the output head can clear the relevant
                # buffers. Note that get_episode_loss(env_index, done=True) doesn't
                # clear the buffers, it just calculates a loss.
                previous_output_head.on_episode_end(env_index_in_previous_batch)

                # Set `done` to `False` for that env, to prevent the output head for the
                # new task from seeing the first observation in the episode as the last.
                observations.done[env_index_in_previous_batch] = False
                # FIXME: If we modify that entry in-place, then even after this method
                # returns, the change will persist.. Therefore we just save the indices
                # that we altered, and reset them before returning.
                done_set_to_false_temporarily_indices.append(env_index_in_previous_batch)
            else:
                raise NotImplementedError(
                    "TODO: The BaseModel doesn't yet support having multiple "
                    "different tasks within the same batch in RL. "
                )
                # IDEA: Need to somehow pass the indices of which env to take care of to
                # each output head, so they can create / clear buffers only when needed.

        assert task_labels is not None
        all_task_indices: Dict[int, Tensor] = get_task_indices(task_labels)

        # Get the loss from each output head:
        if len(all_task_indices) == 1:
            # If everything is in the same task (only one key), no need to split/merge
            # stuff, so it's a bit easier:
            task_id: int = task_labels[0].item()

            self.setup_for_task(task_id)
            # task_output_head = self.output_heads[str(task_id)]
            total_loss += super().output_head_loss(forward_pass, actions=actions, rewards=rewards)
            # total_loss += self.output_head.get_loss(
            #     forward_pass, actions=actions, rewards=rewards,
            # )
        else:
            # Split off the input batch, do a forward pass for each sub-task.
            # (could be done in parallel but whatever.)
            # TODO: Also, not sure if this will play well with DP, DDP, etc.
            for task_id, task_indices in all_task_indices.items():
                # Make a partial observation without the task labels, so that
                # super().forward will use the current output head.
                logger.debug(
                    f"Getting output head loss for "
                    f"{len(task_indices)/batch_size:.0%} of the batch which "
                    f"has task_id of '{task_id}'."
                )

                self.setup_for_task(task_id)
                task_loss = super().output_head_loss(
                    forward_pass=get_slice(forward_pass, task_indices),
                    actions=get_slice(actions, task_indices),
                    rewards=get_slice(rewards, task_indices),
                )
                # NOTE: useful for debugging, but shouldn't be enabled normally.
                # task_loss.name += f"(task {task_id})"
                logger.debug(f"Task {task_id} loss: {task_loss}")
                total_loss += task_loss

        self.previous_task_labels = task_labels
        # FIXME: Reset the 'done' to True, if we manually set it to False.
        for index in done_set_to_false_temporarily_indices:
            observations.done[index] = True

        return total_loss

    def on_before_zero_grad(self, optimizer):
        super().on_before_zero_grad(optimizer)
        from sequoia.methods.models.output_heads.rl import PolicyHead

        for task_id_string, output_head in self.output_heads.items():
            if isinstance(output_head, PolicyHead):
                output_head.detach_all_buffers()

    def shared_step(
        self,
        batch: Tuple[Observations, Optional[Rewards]],
        batch_idx: int,
        environment: Environment,
        phase: str,
        dataloader_idx: int = None,
        optimizer_idx: int = None,
    ) -> Dict:
        assert phase
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
            phase=phase,
            dataloader_idx=dataloader_idx,
            optimizer_idx=optimizer_idx,
        )

    def on_task_switch(self, task_id: Optional[int]):
        """Called when switching between tasks.

        Args:
            task_id (int, optional): the id of the new task. When None, we are
            basically being informed that there is a task boundary, but without
            knowing what task we're switching to.

        NOTE: You can check wether this task switch is occuring at train or test time
        using `self.training`.
        """
        logger.info(f"Switching from task {self.current_task} -> {task_id}.")

        # TODO: Move these to the base model perhaps? (In case there is ever a
        # re-ordering of the mixins that make up the BaseModel)
        super().on_task_switch(task_id)

        self.previous_task = self.current_task
        self.current_task = task_id

        if task_id is not None and self.hp.multihead:
            # Switch the output head to use.
            self.output_head = self.get_or_create_output_head(task_id)

    def shared_modules(self) -> Dict[str, nn.Module]:
        """Returns any trainable modules in `self` that are shared across tasks.

        By giving this information, these weights can then be used in
        regularization-based auxiliary tasks like EWC, for example.

        This dict contains the encoder and output head, by default, as well as any
        shared modules in the auxiliary tasks.

        When using only multiple output heads (i.e. when `self.hp.multihead` is `True`),
        then we remove the output head from the dict before returning it.

        Returns
        -------
        Dict[str, nn.Module]:
            Dictionary mapping from name to the shared modules, if any.
        """
        shared_modules = super().shared_modules()
        if self.hp.multihead:
            shared_modules.pop("output_head")
        return shared_modules

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

        missing_keys, unexpected_keys = super().load_state_dict(state_dict=state_dict, strict=False)

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

    def get_or_create_output_head(self, task_id: int) -> nn.Module:
        """Retrieves or creates a new output head for the given task index.

        Also stores it in the `output_heads`, and adds its parameters to the
        optimizer.
        """
        task_output_head: nn.Module
        assert self.hp.multihead, "This should get called when model isnt multi-headed!"
        if str(task_id) in self.output_heads.keys():
            task_output_head = self.output_heads[str(task_id)]
        else:
            logger.info(f"Creating a new output head for task {task_id}.")
            # NOTE: This also takes care to add the output head's parameters to the
            # optimizer.
            task_output_head = self.create_output_head(task_id=task_id)
            self.output_heads[str(task_id)] = task_output_head
        return task_output_head

    def forward(self, observations: IncrementalAssumption.Observations) -> ForwardPass:
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
        # TODO: Shouldn't have to do this here, since we have the @auto_move_data dec...
        # observations = observations.to(self.device)
        task_ids: Optional[Tensor] = observations.task_labels

        if isinstance(task_ids, np.ndarray) and task_ids.dtype == np.object:
            task_ids = task_ids.tolist()
            if len(task_ids) == 1:
                task_ids = task_ids[0]
        if task_ids is None:
            # Run the forward pass with task inference turned on.
            return self.task_inference_forward_pass(observations)
        task_ids = torch.as_tensor(task_ids, device=self.device, dtype=int)

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

        if task_id != self.current_task and self.hp.multihead:
            # Setup the model for this task. For now we just switch the output head.
            self.output_head = self.get_or_create_output_head(task_id)

        return super().forward(observations)

    def setup_for_task(self, task_id: int) -> None:
        if task_id is not None and self.hp.multihead:
            # Setup the model for this task. For now we just switch the output head.
            self.output_head = self.get_or_create_output_head(task_id)

    def split_forward_pass(self, observations: Observations) -> ForwardPass:
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
        assert self.hp.multihead, "Can only use split forward pass with multiple heads."
        # We have task labels.
        task_labels = observations.task_labels
        if isinstance(task_labels, Tensor):
            task_labels = task_labels.cpu().numpy()

        # Get the indices of the items from each task.
        all_task_indices_dict: Dict[int, np.ndarray] = get_task_indices(task_labels)

        if len(all_task_indices_dict) == 1:
            # No need to split the input, since everything is from the same task.
            task_id: int = task_labels[0].item()
            self.setup_for_task(task_id)
            return self.forward(observations)

        # Placeholder for the predicitons for each item in the batch.
        # NOTE: We put each item in the batch in this list and then stack the results.
        batch_size = len(task_labels)
        task_outputs: List[Batch] = [None for _ in range(batch_size)]

        for task_id, task_indices in all_task_indices_dict.items():
            # Take a slice of the observations, in which all items come from this task.
            task_observations = get_slice(observations, task_indices)
            # Perform a "normal" forward pass (Base case).
            task_output = self.forward(task_observations)

            # Store the outputs for the items from this task in the list.
            for i, index in enumerate(task_indices):
                task_outputs[index] = get_slice(task_output, i)

        # Stack the results.
        assert all(item is not None for item in task_outputs)
        merged_outputs = concatenate(task_outputs)
        return merged_outputs

    def task_inference_forward_pass(self, observations: Observations) -> Tensor:
        """Forward pass with a simple form of task inference."""
        # We don't have access to task labels (`task_labels` is None).
        # --> Perform a simple kind of task inference:
        # 1. Perform a forward pass with each task's output head;
        # 2. Merge these predictions into a single prediction somehow.
        assert observations.task_labels is None or all(observations.task_labels == None)
        # NOTE: This assumes that the observations are batched.
        # These are used below to indicate the shape of the different tensors.
        B = observations.x.shape[0]
        T = n_known_tasks = len(self.output_heads)
        N = self.action_space.n
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
            task_forward_pass = self.forward(task_observations)

            task_outputs[task_id] = task_forward_pass

        # 'Merge' the predictions from each output head using some kind of task
        # inference.
        assert all(item is not None for item in task_outputs)
        # Stack the predictions (logits) from each output head.
        stacked_forward_pass: ForwardPass = stack(task_outputs, dim=1)
        logits_from_each_head = stacked_forward_pass.actions.logits
        assert logits_from_each_head.shape == (B, T, N), (logits_from_each_head.shape, (B, T, N))

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
        selected_forward_pass = stacked_forward_pass[selected_mask]
        assert selected_forward_pass.actions.logits.shape == (B, N)
        return selected_forward_pass


from typing import Dict, Tuple, TypeVar

Dataclass = TypeVar("Dataclass", bound=Batch)


def get_task_indices(
    task_labels: Union[List[Optional[int]], np.ndarray, Tensor]
) -> Dict[Optional[int], Union[np.ndarray, Tensor]]:
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
    all_task_indices: Dict[Optional[int], Union[np.ndarray, Tensor]] = {}

    if task_labels is None:
        return {}

    output_type = np.asarray

    assert isinstance(task_labels, (np.ndarray, Tensor))

    if isinstance(task_labels, Tensor):
        assert task_labels.ndim == 1 or task_labels.size() == 1, task_labels
        task_labels = task_labels.reshape(-1)
    else:
        assert task_labels.ndim == 1 or task_labels.size == 1, task_labels
        task_labels = task_labels.reshape(-1)

    unique_task_labels = list(set(task_labels.tolist()))

    batch_size = len(task_labels)
    # Get the indices for each task.
    for task_id in unique_task_labels:
        if isinstance(task_labels, np.ndarray):
            task_indices = np.arange(batch_size)[task_labels == task_id]
        else:
            assert isinstance(task_labels, Tensor), task_labels
            task_indices = torch.arange(batch_size, device=task_labels.device)[
                task_labels == task_id
            ]
        all_task_indices[task_id] = task_indices
    return all_task_indices


# TODO: Remove this, currently unused.
def cleanup_task_labels(
    task_labels: Optional[Sequence[Optional[int]]],
) -> Optional[np.ndarray]:
    """'cleans up' the task labels, by returning either None or an integer numpy array.

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
            elif not any(task_labels == None):
                task_labels = torch.as_tensor(task_labels.astype(int))
            else:
                raise NotImplementedError(f"TODO: Only given a portion of task labels?")
                # IDEA: Maybe set task_id to -1 in those cases, and return an int
                # ndarray as well?
    if task_labels is None:
        return None
    assert isinstance(task_labels, (np.ndarray, Tensor)), task_labels
    if not task_labels.shape:
        task_labels = task_labels.reshape([1])
    if isinstance(task_labels, Tensor):
        task_labels = task_labels.cpu().numpy()
    if task_labels is not None:
        task_labels = task_labels.astype(int)
    assert task_labels is None or isinstance(task_labels, np.ndarray)
    return task_labels
