from typing import Dict, Type, Union, Tuple

import torch
from torch import nn

from .decoders import CifarDecoder, ImageNetDecoder, MnistDecoder

def get_decoder_class_for_dataset(input_shape: Union[Tuple[int, int, int], torch.Size]) -> Type[nn.Module]:
    assert len(input_shape) == 3, input_shape
    assert input_shape[0] == min(input_shape), f"should be in C, H, W format: {input_shape}"
    shape: Tuple[int, int, int] = tuple(input_shape)  # type: ignore
    h = shape[1]
    if h == 28:
        return MnistDecoder
    elif h == 32:
        return CifarDecoder
    elif h == 224:
        return ImageNetDecoder
    else:
        raise RuntimeError(f"No decoder available for input shape {shape}")
