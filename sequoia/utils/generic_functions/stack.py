
from functools import singledispatch
from typing import Sequence, Union, Dict

import numpy as np
import torch
from gym import Space, spaces
from torch import Tensor
from sequoia.common.spaces import Sparse


@singledispatch
def stack(space: Space,
          items: Sequence,
          out: Union[tuple, dict, Tensor] = None) -> Union[tuple, dict, np.ndarray]:
    raise NotImplementedError(f"Space {space} isn't supported!")


@stack.register(spaces.Box)
@stack.register(spaces.Discrete)
@stack.register(spaces.MultiDiscrete)
@stack.register(spaces.MultiBinary)
def _stack_base(space: Space,
                items: Union[list, tuple],
                out: Union[tuple, dict, Tensor] = None) -> Tensor:
    if None in items:
        return np.array(items)
    try:
        return torch.stack(items, axis=0, out=out)
    except TypeError:
        return np.stack(items, axis=0, out=out)

@stack.register
def _stack_sparse(space: Sparse,
                  items: Union[list, tuple],
                  out: Union[tuple, dict, Tensor] = None) -> Tensor:
    if space.non_prob == 0 or all(item is not None for item in items):
        return stack(space.base, items, out=out)
    else:
        return np.stack(items, out=out)


@stack.register(spaces.Tuple)
def _stack_tuples(space: spaces.Tuple,
                  items: Union[list, tuple],
                  out: Union[tuple, dict, Tensor] = None) -> tuple:
    stacked_items = tuple(
        stack(
            subspace,
            [item[i] for item in items],
            out=(out[i] if out is not None else None),
        )
        for (i, subspace) in enumerate(space.spaces)
    )
    # TODO: Batch isn't imported here. Should instead be done in batch.py if
    # it isn't already.
    # if isinstance(items[0], Batch):
    #     return type(items[0])(*stacked_items)
    return stacked_items


@stack.register(spaces.Dict)
def _stack_dicts(space: spaces.Dict,
                 items: Union[list, tuple],
                 out: Union[tuple, dict, Tensor]) -> Dict:
    return {
        key: stack(subspace, [item[key] for item in items], out=out[key])
        for (key, subspace) in space.spaces.items()
    }


@stack.register(spaces.Space)
def _stack_custom(space: Space,
                  items: Union[list, tuple],
                  out: Union[tuple, dict, Tensor]) -> Union[tuple, dict, Tensor]:
    return tuple(items)

