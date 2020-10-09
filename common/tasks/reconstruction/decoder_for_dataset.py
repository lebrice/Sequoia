from typing import Dict, Type, Union, Tuple

import torch
from torch import nn

from .decoders import CifarDecoder, ImageNetDecoder, MnistDecoder

import functools


# Dict mapping from image (height, width) to the type of decoder to use.
# TODO: Add some more decoders for other image datasets/shapes.
registered_decoders: Dict[Tuple[int, int], Type[nn.Module]] = {
    (28, 28): MnistDecoder,
    (32, 32): CifarDecoder,
    (224, 224): ImageNetDecoder,
}

def get_decoder_class_for_dataset(input_shape: Union[Tuple[int, int, int]]) -> Type[nn.Module]:
    assert len(input_shape) == 3, input_shape
    channels: int
    width: int
    height: int
    if input_shape[0] == min(input_shape):
        # Image is in C, H, W format
        channels, height, width = input_shape
    elif input_shape[-1] == min(input_shape):
        height, width, channels = input_shape
    if (height, width) in registered_decoders:
        return registered_decoders[(height, width)]
    raise RuntimeError(f"No decoder available for input shape {input_shape}")
