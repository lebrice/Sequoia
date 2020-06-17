from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import Flatten  # type: ignore
from torchvision import models

from common.layers import ConvBlock
from config import Config

from .classifier import Classifier
from .pretrained_model import get_pretrained_encoder


class CifarClassifier(Classifier):
    def __init__(self,
                 cifar: int,
                 hparams: Classifier.HParams,
                 config: Config):
        assert cifar in {10, 100}
        self.hidden_size = hparams.hidden_size
        if hparams.encoder_model:
            encoder = get_pretrained_encoder(
                hidden_size=self.hidden_size,
                encoder_model=hparams.encoder_model,
                pretrained=hparams.pretrained_model,
                freeze_pretrained_weights=hparams.freeze_pretrained_model,                
            )
        else:
            print("Using a simple convnet model")
            encoder = nn.Sequential(
                ConvBlock(3, 16, kernel_size=3, padding=1),
                ConvBlock(16, 32, kernel_size=3, padding=1),
                ConvBlock(32, 64, kernel_size=3, padding=1),
                ConvBlock(64, 128, kernel_size=3, padding=1),
                ConvBlock(128, self.hidden_size, kernel_size=3, padding=1),
            )
        super().__init__(
            input_shape=(3,32,32),
            num_classes=cifar,
            encoder=encoder,
            hparams=hparams,
            config=config,
        )
    
    def preprocess_inputs(self, x: Tensor, y: Tensor=None) -> Tuple[Tensor, Optional[Tensor]]:
        # No special preprocessing needed.
        x, y = super().preprocess_inputs(x, y)
        if self.hparams.encoder_model:
            x = normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace=True)
            x = torch.nn.functional.interpolate(x, size=(224, 224))
        return x, y


class Cifar10Classifier(CifarClassifier):
    def __init__(self,
                 hparams: Classifier.HParams,
                 config: Config):
        super().__init__(cifar=10, hparams=hparams, config=config)


class Cifar100Classifier(CifarClassifier):
    def __init__(self,
                 hparams: Classifier.HParams,
                 config: Config):
        super().__init__(cifar=100, hparams=hparams, config=config)


def normalize(tensor, mean, std, inplace=False):
    if not inplace:
        tensor = tensor.clone()
    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    tensor.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
    return tensor
