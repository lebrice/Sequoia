from functools import singledispatch
from typing import Any, Dict, Union, Tuple, Optional

import numpy as np
import torch
from gym import Space, spaces
from torch import Tensor


@singledispatch
def from_tensor(space: Space, sample: Union[Tensor, Any]) -> Union[np.ndarray, Any]:
    """ Converts a Tensor into a sample from the given space. """
    if isinstance(sample, Tensor):
        return sample.cpu().numpy()
    return sample


@from_tensor.register
def _(space: spaces.Discrete, sample: Tensor) -> int:
    if isinstance(sample, Tensor):
        return sample.item()
    return sample


@from_tensor.register
def _(space: spaces.Dict, sample: Dict[str, Union[Tensor, Any]]) -> Dict[str, Union[np.ndarray, Any]]:
    return {
        key: from_tensor(space[key], value)
        for key, value in sample.items()
    }

@from_tensor.register
def _(space: spaces.Tuple, sample: Tuple[Union[Tensor, Any]]) -> Tuple[Union[np.ndarray, Any]]:
    return type(sample)(
        from_tensor(space[i], value)
        for i, value in enumerate(sample)
    )


@singledispatch
def to_tensor(space: Space,
              sample: Union[np.ndarray, Any],
              device: torch.device = None) -> Union[np.ndarray, Any]:
    """ Converts a sample from the given space into a Tensor. """
    return torch.as_tensor(sample, device=device)


from common.spaces.sparse import Sparse


@to_tensor.register(Sparse)
def _(space: Sparse,
      sample: Optional[Any],
      device: torch.device = None) -> Optional[Tensor]:
    if sample is None:
        return None
    return to_tensor(space.base, sample, device)


@to_tensor.register
def _(space: spaces.MultiBinary,
      sample: np.ndarray,
      device: torch.device = None) -> Dict[str, Union[Tensor, Any]]:
    return torch.as_tensor(sample, device=device, dtype=torch.bool)


@to_tensor.register
def _(space: spaces.Dict,
      sample: Dict[str, Union[np.ndarray, Any]],
      device: torch.device = None) -> Dict[str, Union[Tensor, Any]]:
    return {
        key: to_tensor(space[key], value, device)
        for key, value in sample.items()
    }


@to_tensor.register
def _(space: spaces.Tuple,
      sample: Tuple[Union[np.ndarray, Any], ...],
      device: torch.device = None) -> Tuple[Union[Tensor, Any], ...]:
    if len(space.spaces) == 1 and isinstance(space.spaces[0], Sparse):
        # TODO: Debug this, why are we getting to this case?
        if sample == None:
            return (None,)
        assert False, (space, sample)
        
    return type(sample)(
        to_tensor(subspace, sample[i], device)
        for i, subspace in enumerate(space.spaces) 
    )
    
    