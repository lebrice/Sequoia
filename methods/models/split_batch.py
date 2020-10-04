import dataclasses
import functools
from typing import Any, Tuple, Type

from torch import Tensor

from .batch import Batch, Observation, Reward


@functools.lru_cache()
def n_fields(batch_type: Type) -> int:
    return len(dataclasses.fields(batch_type))

@functools.lru_cache()
def n_required_fields(batch_type: Type) -> int:
    # Need to figure out a way to get the number fields through the
    # class itself.
    fields = dataclasses.fields(batch_type)
    required_fields_names = [
        f.name for f in fields
        if f.default is dataclasses.MISSING and
        f.default_factory is dataclasses.MISSING
    ]
    # print(f"class {batch_type}: required fields: {required_fields_names}")
    return len(required_fields_names)


def split_batch(batch: Any, Observation: Type[Batch], Reward: Type[Reward]) -> Tuple[Observation, Reward]:
    """ Split a batch into Observations and Rewards.
    
    To make this simpler, we're always going to return an Observation and a Reward
    object, even if the batch is unlabeled. In that case, the Reward object
    will have `y=None`.
    
    WIP: IDEA: Split the batch intelligently, depending on the
    number of 'required' fields in `Observation` and in `Reward` classes.
    TODO: Would be best to move this method on the Setting somehow.
    """
    if isinstance(batch, Tensor):
        batch = (batch,)
    
    if isinstance(batch, dict):
        obs_fields = Observation.field_names
        rew_fields = Reward.field_names
        assert not set(obs_fields).intersection(set(rew_fields)), (
            "Observation and Reward shouldn't share fields names"
        )
        obs_kwargs = {
            k: v for k, v in batch.items() if k in obs_fields
        }
        obs = Observation(**obs_kwargs)
        reward_kwargs = {
            k: v for k, v in batch.items() if k in rew_fields
        }
        reward = Reward(**reward_kwargs)
        return obs, reward

    if not isinstance(batch, (tuple, list)):
        raise NotImplementedError(f"TODO: support batches of type {type(batch)}")
    
    # If the batch already has two elements, check if they are already of
    # the right type, to avoid unnecessary computation below.
    if len(batch) == 2:
        obs, rew = batch
        if isinstance(obs, Observation) and isinstance(rew, Reward):
            return obs, rew

    # NOTE: Its not a big deal that we call these functions on every batch
    # rather than just once, because they use an LRU cache, and we're
    # probably always calling it with the same argument.

    # Get the min, max and total number of args for each object type.
    min_for_obs = n_required_fields(Observation)
    max_for_obs = n_fields(Observation)
    n_required_for_obs = min_for_obs
    n_optional_for_obs = max_for_obs - min_for_obs
    
    min_for_rew = n_required_fields(Reward)
    max_for_reward = n_fields(Reward)
    n_required_for_rew = min_for_rew
    n_optional_for_rew = max_for_reward - min_for_obs
    
    min_items = min_for_obs + min_for_rew
    max_items = max_for_obs + max_for_reward
    
    n_items = len(batch)
    if  (n_items < min_items or
            n_items > max_items):
        raise RuntimeError(
            f"There aren't the right number of elements in the batch to "
            f"create both an Observation and a Reward!\n"
            f"(batch has {n_items} items, but type "
            f"{Observation} requires from {min_for_obs} to "
            f"{max_for_obs} args, while {Reward} requires from "
            f"{min_for_rew} to {max_for_reward} args. "
        )
    
    # Batch looks like:
    # [
    #     O_1, O_2, ..., O_{min_obs}, (O_{min_obs+1}), ..., (O_{max_obs}),
    #     R_1, R_2, ..., R_{min_rew}, (R_{min_rew+1}), ..., (R_{max_rew}),
    # ]
    if n_items == 0:
        obs = Observation()
        rew = Reward()
    if n_items == max_items:
        # Easiest case! Just use all the values.
        obs = Observation(*batch[:max_for_obs])
        rew = Reward(*batch[max_for_obs:])
    elif n_items == min_items:
        # Easy case as well. Also simply uses all the values directly.
        obs = Observation(*batch[:min_for_obs])
        rew = Reward(*batch[min_for_obs:])
    elif n_optional_for_obs == 0 and n_optional_for_rew != 0:
        # All the extra args go in the reward.
        obs = Observation(*batch[:min_for_obs])
        rew = Reward(*batch[min_for_obs:])
    elif n_optional_for_obs != 0 and n_optional_for_rew == 0:
        # All the extra args go in the observation.
        obs = Observation(*batch[:max_for_obs])
        rew = Reward(*batch[max_for_obs:])
    else:
        # We can't tell where the 'extra' tensors should go.
        
        # TODO: Maybe just assume that all the 'extra' tensors are meant to
        # be part of the observation? or the reward? For instance:
        # Option 1: All the extra args go in the observation:
        # obs = Observation(*batch[:n_items-n_required_for_rew])
        # rew = Observation(*batch[n_items-n_required_for_rew:])
        # Option 2: All the extra args go in the reward:
        # obs = Observation(*batch[:n_required_for_obs])
        # rew = Observation(*batch[n_required_for_obs:])
        n_extra = n_items - min_items
        max_extra = n_optional_for_obs + n_optional_for_rew
        raise NotImplementedError(
            f"Can't tell where to put these extra tensors!\n"
            f"(batch has {n_items} items, but type "
            f"{Observation} requires from {min_for_obs} to "
            f"{max_for_obs} args, while {Reward} requires from "
            f"{min_for_rew} to {max_for_reward} args. There are "
            f"{n_extra} extra items out of a potential of {max_extra}."
        )
    return obs, rew
