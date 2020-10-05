from .batch_transform import BatchTransform
from dataclasses import dataclass
from .channels import ChannelsFirstIfNeeded
from common import Batch 
import torch
from torch import Tensor

@dataclass(frozen=True)
class Obs(Batch):
    x: Tensor


def test_transform_applies_also_on_batch_object():
    x = torch.ones([10, 28, 28, 3], names=["N", "H", "W", "C"])
    transform = ChannelsFirstIfNeeded()
    
    # Add the BatchTransform base class if not present:
    if not isinstance(transform, BatchTransform):    
        class NewVersion(ChannelsFirstIfNeeded, BatchTransform):
            pass
        transform = NewVersion()

    obs = Obs(x)
    
    # Get the result of applying the transform on the tensor directly.
    x_result = transform(x)
    # Get the result of applying the transform on the Batch object.
    result: Obs = transform(obs)
    assert isinstance(result, Obs)

    # Expect the `x` tensor to be the same as if the transform was applied on it.
    expected_result = Obs(x_result)
    assert result.x.tolist() == expected_result.x.tolist()