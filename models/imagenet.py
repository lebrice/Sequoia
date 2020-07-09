from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torchvision import models
from torchvision.models import resnet50

from common.layers import ConvBlock
from config import Config

from .classifier import Classifier
from .pretrained_model import get_pretrained_encoder


class ImageNetClassifier(Classifier):

    def __init__(self, hparams: Classifier.HParams, config: Config):
        super().__init__(hparams=hparams, config=config)

