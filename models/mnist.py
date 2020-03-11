from torch import nn

from common.layers import ConvBlock, Flatten
from config import Config

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
            nn.Linear(self.hidden_size, self.hidden_size//2),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(self.hidden_size//2, 10),
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
