from gym.vector.utils import (
    batch_space as _batch_space,
    concatenate as _concatenate,
    create_shared_memory as _create_shared_memory,
    create_empty_array as _create_empty_array
)
from gym.spaces import Space
from functools import singledispatch
import multiprocessing as mp
import numpy as np
from typing import Dict, Type, Optional

# A map that gives the dtype to use for the result of stacking/batching items of a given type. 
_item_dtype_to_batch_dtype_map: Dict[Type, Type] = {}

def get_batch_type_for_item_type(item_type: Type) -> Optional[Type]:
    """ Returns the dtype to use for 'batches' of items of type `item_type`.
    
    When no type is registered, returns None.
    """
    return _item_dtype_to_batch_dtype_map.get(item_type)

def get_item_type_for_batch_type(batch_type: Type) -> Optional[Type]:
    """ Returns the dtype to use for the items of a 'batch' of type `batch_type`.

    When no type is registered, returns None. If there is more than one item type for the given
    batch type, returns None.
    """
    matches =  []
    for item_type, b_type in _item_dtype_to_batch_dtype_map.items():
        if b_type is batch_type:
            matches.append(item_type)
    # NOTE: This shouldn't happen if the types are registered using the right function below:
    assert len(matches) <= 1, f"More than one item type for batch type {batch_type}: {matches}"
    return matches[0] if matches else None


def register_batch_type_to_use_for_item_type(item_type: Type, batch_type: Type) -> None:
    for i_type, b_type in _item_dtype_to_batch_dtype_map.items():
        if i_type is item_type and b_type is batch_type:
            # Already registered, all good.
            return
        if i_type is item_type:
            raise RuntimeError(
                f"There is already a different batch type ({b_type}) registered for item type "
                f"{i_type}!"
            )
        if b_type is batch_type:
            raise RuntimeError(
                f"There is already an item type ({i_type}) registerd for batch type {b_type}!"
            )
    _item_dtype_to_batch_dtype_map[item_type] = batch_type


@singledispatch
def batch_space(space: Space, n: int) -> Space:
    return _batch_space(space, n=n)


@singledispatch
def concatenate(space: Space, items, out):
    _concatenate(items, out, space)


@singledispatch
def create_shared_memory(space, n=1, ctx=mp):
    return _create_shared_memory(space, n=n, ctx=ctx)


@singledispatch
def create_empty_array(space, n=1, fn=np.zeros):
    return _create_empty_array(space, n=n, fn=fn)
