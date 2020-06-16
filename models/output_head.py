from torch import nn, Tensor
from dataclasses import dataclass
from typing import List
from simple_parsing import list_field
from utils.json_utils import Serializable
from torch.nn import Flatten


class OutputHead(nn.Module):
    """Module for the output head of the model.
    
    NOTE: Just a simple dense block for now.
    TODO: Add the invertible networks or other cool tricks as subclasses maybe.
    """

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

    
    def __init__(self, input_size: int, output_size: int, hparams: "OutputHead.HParams"=None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hparams = hparams or self.HParams()

        hidden_layers: List[nn.Module] = []
        in_features = self.input_size
        for i, neurons in enumerate(self.hparams.hidden_neurons):
            out_features = neurons
            hidden_layers.append(nn.Linear(in_features, out_features))
            in_features = out_features # next input size is output size of prev.
        
        self.flatten = Flatten()
        self.dense = nn.Sequential(*hidden_layers)
        self.output = nn.Linear(in_features, output_size)

    def forward(self, h_x: Tensor) -> Tensor:  # type: ignore
        h_x = self.flatten(h_x)
        x = self.dense(h_x)
        return self.output(x)
