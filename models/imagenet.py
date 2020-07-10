from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torchvision import models
from torchvision.models import resnet50

from common.layers import ConvBlock
from config import Config
from datasets import Datasets, DatasetConfig

from .classifier import Classifier
from .pretrained_model import get_pretrained_encoder


class ImageNetClassifier(Classifier):
    def __init__(self,
                 hparams: Classifier.HParams,
                 config: Config,
                 dataset_config: DatasetConfig = None):
        self.hidden_size = hparams.hidden_size
        self.dataset_config = Datasets.imagenet.value if dataset_config is None else dataset_config.value
        # We use a Resnet50 by default encoder by default.
        hparams.encoder_model = hparams.encoder_model or resnet50

        encoder = get_pretrained_encoder(
            hidden_size=self.hidden_size,   
            encoder_model=hparams.encoder_model,
            pretrained=hparams.pretrained_model,
            freeze_pretrained_weights=hparams.freeze_pretrained_model
        )
        super().__init__(
            input_shape=self.dataset_config.x_shape,
            num_classes=self.dataset_config.num_classes,
            encoder=encoder,
            hparams=hparams,
            config=config,
        )


def normalize(tensor: Tensor, mean, std, inplace=False) -> Tensor:
    if not inplace:
        tensor = tensor.clone()
    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std  = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    tensor.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
    return tensor

class MiniImageNetClassifier(ImageNetClassifier):
    def __init__(self,
                 hparams: Classifier.HParams,
                 config: Config, 
                 dataset_config: DatasetConfig = None):
        
         super().__init__(hparams,config,Datasets.mini_imagenet)
