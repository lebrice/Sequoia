from typing import Dict, Type, Union, Tuple

import torch
from torch import nn

from datasets.datasets import Datasets

from .decoders import CifarDecoder, ImageNetDecoder, MnistDecoder

# NOTE: Can't use the dataset as the key, since the auxiliary tasks don't
# currently have access to that information, they only get access to the input
# shape.

decoder_class_for_input_shape: Dict[Tuple[int, int, int], Type[nn.Module]] = {
    Datasets.mnist.value.x_shape: MnistDecoder,
    Datasets.fashion_mnist.value.x_shape: MnistDecoder,
    Datasets.cifar10.value.x_shape: CifarDecoder,
    # Datasets.cifar100.value.x_shape: CifarDecoder,
    Datasets.imagenet.value.x_shape: ImageNetDecoder,
}

def get_decoder_class_for_dataset(input_shape: Union[Tuple[int, int, int], torch.Size]) -> Type[nn.Module]:
    assert len(input_shape) == 3
    shape: Tuple[int, int, int] = tuple(input_shape)  # type: ignore
    if shape not in decoder_class_for_input_shape:
        raise RuntimeError(f"No decoder available for input shape {shape}")
    return decoder_class_for_input_shape[shape]
