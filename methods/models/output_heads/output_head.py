from dataclasses import dataclass
from typing import ClassVar, List

from torch import Tensor, nn
from torch.nn import Flatten  # type: ignore

from common.loss import Loss
from common.metrics import ClassificationMetrics
from simple_parsing import list_field
from utils.json_utils import Serializable
from utils.utils import camel_case, remove_suffix


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

    def __init__(self, input_size: int, output_size: int, hparams: "OutputHead.HParams" = None, name: str = ""):
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
            in_features = out_features # next input size is output size of prev.
        
        self.flatten = Flatten()
        self.dense = nn.Sequential(*hidden_layers)
        self.output = nn.Linear(in_features, output_size)

        # For example, but you could change this in your derived class.
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, h_x: Tensor) -> Tensor:  # type: ignore
        h_x = self.flatten(h_x)
        x = self.dense(h_x)
        return self.output(x)

    def get_loss(self, x: Tensor, h_x: Tensor, y_pred: Tensor, y: Tensor) -> Loss:
        loss = self.loss_fn(y_pred, y)
        assert self.name, "Output Heads should have a name!"
        loss_info = Loss(
            name=self.name,
            loss=loss,
            # NOTE: we're passing the tensors to the Loss object because we let
            # it create the Metrics for us automatically.
            x=x,
            h_x=h_x,
            y_pred=y_pred,
            y=y,
        )
        return loss_info
