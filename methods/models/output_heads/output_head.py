from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Dict, List, Any

from torch import Tensor, nn
from torch.nn import Flatten  # type: ignore

from common.loss import Loss
from common.metrics import get_metrics, ClassificationMetrics
from simple_parsing import list_field
from utils.serialization import Serializable
from utils.utils import camel_case, remove_suffix

from settings import Observations, Actions, Rewards
from ..forward_pass import ForwardPass 


class OutputHead(nn.Module):
    """Module for the output head of the model.
    
    This output head is meant for classification, but you could inherit from it
    and customize it for doing something different like RL or reconstruction, 
    for instance.
    """
    # TODO: Rename this to 'output' and create some ClassificationHead,
    # RegressionHead, ValueHead, etc. subclasses with the corresponding names.
    name: ClassVar[str] = "classification"

    @dataclass
    class HParams(Serializable):
        """ Hyperparameters of the output head. """
        # Number of hidden layers in the output head.
        hidden_layers: int = 0
        # Number of neurons in each hidden layer of the output head.
        # If a single value is given, than each of the `hidden_layers` layers
        # will have that number of neurons. 
        # If `n > 1` values are given, then `hidden_layers` must either be 0 or
        # `n`, otherwise a RuntimeError will be raised.
        hidden_neurons: List[int] = list_field(128)

        def __post_init__(self):
            # no value passed to --hidden_layers
            if self.hidden_layers == 0:
                if len(self.hidden_neurons) == 1:
                    # Default Setting: No hidden layers.
                    self.hidden_neurons = []
                elif len(self.hidden_neurons) > 1:
                    # Set the number of hidden layers to the number of passed values.
                    self.hidden_layers = len(self.hidden_neurons)
            elif self.hidden_layers > 0 and len(self.hidden_neurons) == 1:
                # Duplicate that value for each of the `hidden_layers` layers.
                self.hidden_neurons *= self.hidden_layers
            if self.hidden_layers != len(self.hidden_neurons):
                raise RuntimeError(
                    f"Invalid values: hidden_layers ({self.hidden_layers}) != "
                    f"len(hidden_neurons) ({len(self.hidden_neurons)})."
                )

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 hparams: "OutputHead.HParams" = None,
                 name: str = ""):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hparams = hparams or self.HParams()
        self.name = name or type(self).name

        hidden_layers: List[nn.Module] = []
        in_features = self.input_size
        for i, neurons in enumerate(self.hparams.hidden_neurons):
            out_features = neurons
            hidden_layers.append(nn.Linear(in_features, out_features))
            hidden_layers.append(nn.ReLU())
            in_features = out_features # next input size is output size of prev.

        self.dense = nn.Sequential(
            Flatten(),
            *hidden_layers,
            nn.Linear(in_features, output_size)
        )

        # For example, but you could change this in your subclass.
        self.loss_fn = nn.CrossEntropyLoss()

    @abstractmethod
    def forward(self, observations: Observations, representations: Tensor) -> Actions:
        """Given the observations and their representations, produce "actions".
        
        Parameters
        ----------
        observations : Observations
            Object containing the input examples.
        representations : Any
            The results of encoding the input examples.

        Returns
        -------
        Actions
            An object containing the action to take, and which can be used to
            calculate the loss later on.
        """
        # TODO: This should probably take in a dict and return a dict, or something like that?
        return self.dense(representations)

    @abstractmethod
    def get_loss(self, forward_pass: ForwardPass, y: Tensor) -> Loss:
        """ Given the forward pass (including the actions produced by this
        output head), and the corresponding rewards, get a Loss to use for
        training.
        """
        
