from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type, ClassVar

import torchvision.models as tv_models
from simple_parsing import choice, mutable_field
from torch import nn, optim
from torch.optim.optimizer import Optimizer  # type: ignore

from methods.models.output_heads import OutputHead
from utils import Parseable, Serializable

from .pretrained_utils import get_pretrained_encoder


available_optimizers: Dict[str, Type[Optimizer]] = {
    "sgd": optim.SGD,
    "adam": optim.Adam,
    "rmsprop": optim.RMSprop,
}

available_encoders: Dict[str, Type[nn.Module]] = {
    "vgg16": tv_models.vgg16,
    "resnet18": tv_models.resnet18,
    "resnet34": tv_models.resnet34,
    "resnet50": tv_models.resnet50,
    "resnet101": tv_models.resnet101,
    "resnet152": tv_models.resnet152,
    "alexnet": tv_models.alexnet,
    "densenet": tv_models.densenet161,
    # TODO: Add the self-supervised pl modules here!
}

@dataclass
class BaseHParams(Serializable, Parseable):
    """ Set of 'base' Hyperparameters for the 'base' LightningModule. """
    # Class variable versions of the above dicts, for easier subclassing.
    # NOTE: These don't get parsed from the command-line.
    available_optimizers: ClassVar[Dict[str, Type[Optimizer]]] = available_optimizers
    available_encoders: ClassVar[Dict[str, Type[nn.Module]]] = available_optimizers
    
    # Batch size to use.
    # TODO: Would we need to change this when using DP or DDP of
    # pytorch-lightning?
    batch_size: int = 64

    # Number of hidden units (before the output head).
    # When left to None (default), the hidden size from the pretrained
    # encoder model will be used. When set to an integer value, an
    # additional Linear layer will be placed between the outputs of the
    # encoder in order to map from the pretrained encoder's output size H_e
    # to this new hidden size `new_hidden_size`.
    new_hidden_size: Optional[int] = None
    # Which optimizer to use.
    optimizer: str = choice(available_optimizers.keys(), default="adam")
    # Learning rate of the optimizer.
    learning_rate: float = 0.001
    # L2 regularization term for the model weights.
    weight_decay: float = 1e-6
    # Use an encoder architecture from the torchvision.models package.
    encoder: str = choice(available_encoders.keys(), default="resnet18")
    # Retrain the encoder from scratch.
    train_from_scratch: bool = False
    # Wether we should keep the weights of the pretrained encoder frozen.
    freeze_pretrained_encoder_weights: bool = False

    # Settings for the output head.
    # TODO: This could be overwritten in a subclass to do classification or
    # regression or RL, etc.
    output_head: OutputHead.HParams = mutable_field(OutputHead.HParams)

    # Wether the output head should be detached from the representations.
    # In other words, if the gradients from the downstream task should be
    # allowed to affect the representations.
    detach_output_head: bool = False

    def __post_init__(self):
        """Use this to initialize (or fix) any fields parsed from the
        command-line.
        """

    def make_optimizer(self, *args, **kwargs) -> Optimizer:
        """ Creates the Optimizer object from the options. """
        optimizer_class = self.available_optimizers[self.optimizer]
        options = {
            "lr": self.learning_rate,
            "weight_decay": self.weight_decay,
        }
        options.update(kwargs)
        return optimizer_class(*args, **options)

    def make_encoder(self) -> Tuple[nn.Module, int]:
        """Creates an Encoder model and returns the resulting hidden size.

        Returns:
            Tuple[nn.Module, int]: the encoder and the hidden size.
        """
        encoder_model = self.available_encoders[self.encoder]
        encoder, hidden_size = get_pretrained_encoder(
            encoder_model=encoder_model,
            pretrained=not self.train_from_scratch,
            freeze_pretrained_weights=self.freeze_pretrained_encoder_weights,
            new_hidden_size=self.new_hidden_size,
        )
        return encoder, hidden_size

