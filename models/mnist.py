import torch
from torch import nn
from torchvision import models

from common.layers import ConvBlock
from config import Config
from torch.nn import Flatten
from .classifier import Classifier


class MnistClassifier(Classifier):
    def __init__(self,
                 hparams: Classifier.HParams,
                 config: Config):
        self.hidden_size = hparams.hidden_size
        encoder = nn.Sequential(
            ConvBlock(1, 16, kernel_size=3, padding=1),
            ConvBlock(16, 32, kernel_size=3, padding=1),
            ConvBlock(32, self.hidden_size, kernel_size=3, padding=1),
            ConvBlock(self.hidden_size, self.hidden_size, kernel_size=3, padding=1),
        )
        classifier = nn.Sequential(
            Flatten(),
            nn.Linear(self.hidden_size, 10),
        )
        super().__init__(
            input_shape=(1,28,28),
            num_classes=10,
            encoder=encoder,
            classifier=classifier,
            hparams=hparams,
            config=config,
        )
    
    def preprocess_inputs(self, x):
        # No special preprocessing needed.
        return super().preprocess_inputs(x)


class MnistPretrainedEncoderClassifier(Classifier):
    def __init__(self,
                 hparams: Classifier.HParams,
                 config: Config):
        self.hidden_size = hparams.hidden_size

        encoder = models.vgg16(pretrained=True)
        for param in encoder.parameters():
            param.requires_grad = False
        encoder.classifier = nn.Linear(25088, self.hidden_size)

        classifier = nn.Sequential(
            Flatten(),
            nn.Linear(self.hidden_size, 10),
        )
        super().__init__(
            input_shape=(1,28,28),
            num_classes=10,
            encoder=encoder,
            classifier=classifier,
            hparams=hparams,
            config=config,
        )
    
    def preprocess_inputs(self, x):
        # No special preprocessing needed.
        x = super().preprocess_inputs(x)
        x = x.repeat(1, 3, 1, 1) # grayscale to rgb
        x = normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace=True)
        return torch.nn.functional.interpolate(x, size=(224, 224))


def normalize(tensor, mean, std, inplace=False):
    if not inplace:
        tensor = tensor.clone()
    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    tensor.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
    return tensor
