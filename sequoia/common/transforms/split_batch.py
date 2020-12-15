import dataclasses
import functools
from typing import Any, Callable, Optional, Tuple, Type, TypeVar

import numpy as np
from torch import Tensor

from ..batch import Batch
from .transform import Transform

# Type variables just for the below function.
ObservationType = TypeVar("ObservationType", bound=Batch)
RewardType = TypeVar("RewardType", bound=Batch)


class SplitBatch(Transform[Any, Tuple[ObservationType, RewardType]]):
    """
    Transform that will split batches into Observations and Rewards.
    
    The provided observation and reward types (which have to be subclasses of
    the `Batch` class) will be used to construct the observation and reward
    objects, respectively.
    
    To make this simpler, this callable will always return an Observation and a
    Reward object, even when the batch is unlabeled. In that case, the Reward
    object will have a 'None' passed for any of its required arguments.

    Parameters
    ----------
    observation_type : Type[ObservationType]
        [description]
    reward_type : Type[RewardType]
        [description]

    Returns
    -------
    Callable[[Any], Tuple[ObservationType, RewardType]]
        [description]

    Raises
    ------
    RuntimeError
        If the observation_type or reward_type don't both subclass Batch.
    NotImplementedError
        If the type of the batch isn't supported.
    RuntimeError
        [description]
    NotImplementedError
        [description]
    """
    def __init__(self,
                 observation_type: Type[ObservationType],
                 reward_type: Type[RewardType]):
        self.Observations = observation_type
        self.Rewards = reward_type
        self.func = split_batch(observation_type=observation_type,
                                reward_type=reward_type)

    def __call__(self, batch: Any) -> Tuple[ObservationType, RewardType]:
        return self.func(batch)


def split_batch(observation_type: Type[ObservationType],
                reward_type: Type[RewardType]) -> Callable[[Any], Tuple[ObservationType,
                                                                        Optional[RewardType]]]:
    """Makes a callable that will split batches into Observations and Rewards.
    
    The provided observation and reward types (which have to be subclasses of
    the `Batch` class) will be used to construct the observation and reward
    objects, respectively.
    
    To make this simpler, this callable will always return a tuple with an
    Observation and an optional Reward object, even when the batch is unlabeled.
    In that case, the Reward will be None.

    Parameters
    ----------
    observation_type : Type[ObservationType]
        [description]
    reward_type : Type[RewardType]
        [description]

    Returns
    -------
    Callable[[Any], Tuple[ObservationType, RewardType]]
        [description]

    Raises
    ------
    RuntimeError
        If the observation_type or reward_type don't both subclass Batch.
    NotImplementedError
        If the type of the batch isn't supported.
    RuntimeError
        [description]
    NotImplementedError
        [description]
    """
    if not (issubclass(observation_type, Batch) and
            issubclass(reward_type, Batch)):
        raise RuntimeError("Both `observation_type` and `reward_type` need to "
                           "inherit from `Batch`!")
    
    # Get the min, max and total number of args for each object type.
    min_for_obs = n_required_fields(observation_type)
    max_for_obs = n_fields(observation_type)
    n_required_for_obs = min_for_obs
    n_optional_for_obs = max_for_obs - min_for_obs
    
    min_for_rew = n_required_fields(reward_type)
    max_for_reward = n_fields(reward_type)
    n_required_for_rew = min_for_rew
    n_optional_for_rew = max_for_reward - min_for_obs
    
    min_items = min_for_obs + min_for_rew
    max_items = max_for_obs + max_for_reward
    
    def split_batch_transform(batch: Any) -> Tuple[ObservationType, RewardType]:
        if isinstance(batch, (Tensor, np.ndarray)):
            batch = (batch,)
        
        if isinstance(batch, dict):
            obs_fields = observation_type.field_names
            rew_fields = reward_type.field_names
            assert not set(obs_fields).intersection(set(rew_fields)), (
                "Observation and Reward shouldn't share fields names"
            )
            obs_kwargs = {
                k: v for k, v in batch.items() if k in obs_fields
            }
            obs = observation_type(**obs_kwargs)
            reward_kwargs = {
                k: v for k, v in batch.items() if k in rew_fields
            }
            reward = reward_type(**reward_kwargs)
            return obs, reward

        if isinstance(batch, observation_type):
            return batch, None

        if not isinstance(batch, (tuple, list)):
            # TODO: Add support for more types maybe? Or just wrap it in a tuple
            # and call it a day?
            raise RuntimeError(f"Batch is of an unsuported type: {type(batch)}.")
        
        # If the batch already has two elements, check if they are already of
        # the right type, to avoid unnecessary computation below.
        if len(batch) == 2:
            obs, rew = batch
            if isinstance(obs, observation_type) and isinstance(rew, reward_type):
                return obs, rew

        n_items = len(batch)
        if  (n_items < min_items or
                n_items > max_items):
            raise RuntimeError(
                f"There aren't the right number of elements in the batch to "
                f"create both an Observation and a Reward!\n"
                f"(batch has {n_items} items, but type "
                f"{observation_type} requires from {min_for_obs} to "
                f"{max_for_obs} args, while {reward_type} requires from "
                f"{min_for_rew} to {max_for_reward} args. "
            )

        # Batch looks like:
        # [
        #     O_1, O_2, ..., O_{min_obs}, (O_{min_obs+1}), ..., (O_{max_obs}),
        #     R_1, R_2, ..., R_{min_rew}, (R_{min_rew+1}), ..., (R_{max_rew}),
        # ]
        if n_items == 0:
            obs = observation_type()
            rew = reward_type()
        if n_items == max_items:
            # Easiest case! Just use all the values.
            obs = observation_type(*batch[:max_for_obs])
            rew = reward_type(*batch[max_for_obs:])
        elif n_items == min_items:
            # Easy case as well. Also simply uses all the values directly.
            obs = observation_type(*batch[:min_for_obs])
            rew = reward_type(*batch[min_for_obs:])
        elif n_optional_for_obs == 0 and n_optional_for_rew != 0:
            # All the extra args go in the reward.
            obs = observation_type(*batch[:min_for_obs])
            rew = reward_type(*batch[min_for_obs:])
        elif n_optional_for_obs != 0 and n_optional_for_rew == 0:
            # All the extra args go in the observation.
            obs = observation_type(*batch[:max_for_obs])
            rew = reward_type(*batch[max_for_obs:])
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
                f"{observation_type} requires from {min_for_obs} to "
                f"{max_for_obs} args, while {reward_type} requires from "
                f"{min_for_rew} to {max_for_reward} args. There are "
                f"{n_extra} extra items out of a potential of {max_extra}."
            )
        return obs, rew

    return split_batch_transform


def n_fields(batch_type: Type[Batch]) -> int:
    """Helper function, gives back the total number of fields in Batch subclass.

    Parameters
    ----------
    batch_type : Type
        A subclass of Batch.

    Returns
    -------
    int
        The total number of fields in the type. See the `fields` function of the
        `dataclasses` package for more info.
    """
    return len(dataclasses.fields(batch_type))


def n_required_fields(batch_type: Type) -> int:
    """Helper function, gives the number of required fields in the dataclass.

    Parameters
    ----------
    batch_type : Type
        [description]

    Returns
    -------
    int
        The number of fields which don't have a default value or a default
        factory and are required by the constructor (have init=True).
    """
    # Need to figure out a way to get the number fields through the
    # class itself.
    fields = dataclasses.fields(batch_type)
    required_fields_names = [
        f.name for f in fields
        if f.default is dataclasses.MISSING and
        f.default_factory is dataclasses.MISSING and
        f.init
    ]
    # print(f"class {batch_type}: required fields: {required_fields_names}")
    return len(required_fields_names)
