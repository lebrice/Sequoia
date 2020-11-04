from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Dict, List, Any

import gym
from gym import spaces
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
                 action_space: gym.Space,
                 reward_space: gym.Space = None,
                 hparams: "OutputHead.HParams" = None,
                 name: str = ""):
        super().__init__()
        self.input_size = input_size
        self.action_space = action_space
        self.reward_space = reward_space or action_space
        self.hparams = hparams or self.HParams()
        self.name = name or type(self).name

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

    @abstractmethod
    def get_loss(self, forward_pass: ForwardPass, y: Tensor) -> Loss:
        """ Given the forward pass (including the actions produced by this
        output head), and the corresponding rewards, get a Loss to use for
        training.
        """
        
