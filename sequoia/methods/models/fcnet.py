""" TODO: Take out the dense network from the OutputHead. """
from torch import nn
from typing import ClassVar, Dict, Type, List
from simple_parsing import list_field
from dataclasses import dataclass
from sequoia.utils import Parseable, Serializable
from sequoia.common.hparams import HyperParameters, uniform, log_uniform, categorical
from typing import overload, Union, Optional

class FCNet(nn.Sequential):
    """ Fully-connected network. """

    @dataclass
    class HParams(HyperParameters):
        """ Hyper-parameters of a fully-connected network. """
        
        available_activations: ClassVar[Dict[str, Type[nn.Module]]] = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "elu": nn.ELU, # No idea what these do, but hey, they are available!
            "gelu": nn.GELU,
            "relu6": nn.ReLU6,
        }
        # Number of hidden layers in the output head.
        hidden_layers: int = uniform(0, 10, default=3)
        # Number of neurons in each hidden layer of the output head.
        # If a single value is given, than each of the `hidden_layers` layers
        # will have that number of neurons. 
        # If `n > 1` values are given, then `hidden_layers` must either be 0 or
        # `n`, otherwise a RuntimeError will be raised.
        hidden_neurons: Union[int, List[int]] = uniform(16, 512, default=64)
        activation: Type[nn.Module] = categorical(available_activations, default=nn.Tanh)
        # Dropout probability. Dropout is applied after each layer.
        # Set to None or 0 for no dropout.
        # TODO: Not sure if this is how it's typically used. Need to check.
        dropout_prob: Optional[float] = uniform(0, 0.8, default=0.2)

        def __post_init__(self):
            super().__post_init__()
            if isinstance(self.activation, str):
                self.activation = self.available_activations[self.activation.lower()]

            if isinstance(self.hidden_neurons, int):
                self.hidden_neurons = [self.hidden_neurons]

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
            elif self.hidden_layers == 1 and not self.hidden_neurons:
                self.hidden_layers = 0

            if self.hidden_layers != len(self.hidden_neurons):
                raise RuntimeError(
                    f"Invalid values: hidden_layers ({self.hidden_layers}) != "
                    f"len(hidden_neurons) ({len(self.hidden_neurons)})."
                )

    @overload
    def __init__(self, in_features: int, out_features: int, hparams: HParams=None): ...
        
    @overload
    def __init__(self, in_features: int, out_features: int, hidden_layers: int=1, hidden_neurons: List[int]=None, activation: Type[nn.Module]=nn.Tanh): ...
    
    def __init__(self, in_features: int, out_features: int, hparams: HParams=None, **kwargs):
        self.in_features = in_features
        self.out_features = out_features
        self.hparams = hparams or self.HParams(**kwargs)
        hidden_layers: List[nn.Module] = []
        output_size = out_features
        assert isinstance(self.hparams.hidden_neurons, list)
        for i, neurons in enumerate(self.hparams.hidden_neurons):
            out_features = neurons
            if self.hparams.dropout_prob:
                hidden_layers.append(nn.Dropout(p=self.hparams.dropout_prob))
            hidden_layers.append(nn.Linear(in_features, out_features))
            hidden_layers.append(self.hparams.activation())
            in_features = out_features # next input size is output size of prev.
        super().__init__(
            nn.Flatten(),
            *hidden_layers,
            nn.Linear(in_features, output_size)
        )

    # TODO: IDEA: use @singledispatchmethod to add a `forward` implementation
    # for mapping input space to output space.
    # def forward(self, input: Any)
