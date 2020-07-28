import torch
import torch.nn.functional as F
from torch import nn, Tensor
from dataclasses import dataclass
from typing import List
from simple_parsing import list_field
from utils.json_utils import Serializable
from torch.nn import Flatten  # type: ignore

class OutputHead(nn.Module):
    @dataclass
    class HParams(Serializable):
        pass

    def __init__(self, input_size: int, output_size: int, hparams: "OutputHead.HParams"=None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hparams = hparams or self.HParams()


class LinearOutputHead(OutputHead):
    """Module for the output head of the model.
    
    NOTE: Just a simple dense block for now.
    TODO: Add the invertible networks or other cool tricks as subclasses maybe.
    """

    @dataclass
    class HParams(OutputHead.HParams):
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
        super().__init__(input_size,output_size,hparams)
        self.hparams = hparams or self.HParams()
        hidden_layers: List[nn.Module] = []
        in_features = self.input_size
        #self.relu = nn.ReLU()
        for i, neurons in enumerate(self.hparams.hidden_neurons):
            out_features = neurons
            hidden_layers.append(nn.Linear(in_features, out_features))
            hidden_layers.append(nn.ReLU())
            in_features = out_features # next input size is output size of prev.
        
        self.flatten = Flatten() 
        self.dense = nn.Sequential(*hidden_layers)
        self.output = nn.Linear(in_features, output_size)

    def forward(self, h_x: Tensor) -> Tensor:  # type: ignore
        x = self.flatten(h_x)
        #x = self.relu(h_x)
        x = self.dense(x)
        return self.output(x)


class OutputHead_DUQ(OutputHead):
    #https://arxiv.org/pdf/2003.02037.pdf
    """Module for the output head of the model.
    """

    @dataclass
    class HParams(OutputHead.HParams):
        """ Distance based output head. """
        centroid_size: int = 512 #Size to use for centroids (default: 512)
        length_scale: float = 0.1 #Length scale of RBF kernel (default: 0.1)",
        gamma:float = 0.999 #Decay factor for exponential average (default: 0.999)",

    
    def __init__(self, input_size: int, output_size: int, hparams: "OutputHead.HParams"=None):
        super().__init__(input_size, output_size, hparams)
        self.hparams = hparams or self.HParams()

        self.x_emb_upd = None
        self.y_emb_upd = None
        self.W = nn.Parameter(
            torch.zeros(self.hparams.centroid_size, self.output_size, self.input_size)
        )
        nn.init.kaiming_normal_(self.W, nonlinearity="relu")

        self.register_buffer("N", torch.zeros(self.output_size) + 13)
        self.register_buffer(
            "m", torch.normal(torch.zeros(self.hparams.centroid_size, self.output_size), 0.05)
        )
        self.m = self.m * self.N
        self.sigma = self.hparams.length_scale


    def rbf(self, z):
        z = torch.einsum("ij,mnj->imn", z, self.W)

        embeddings = self.m / self.N.unsqueeze(0)

        diff = z - embeddings.unsqueeze(0)
        diff = (diff ** 2).mean(1).div(2 * self.sigma ** 2).mul(-1).exp()

        return diff
    def prepare_embedings_update(self, x, y):
        self.x_emb_upd = x
        self.y_emb_upd = y
    
    def update_embeddings(self, encoder, x=None, y=None):
        if x is None:
            x = self.x_emb_upd
            y = self.y_emb_upd
        y = F.one_hot(y, self.output_size).float()
        self.N = self.hparams.gamma * self.N + (1 - self.hparams.gamma) * y.sum(0)

        z = encoder(x)
        #print(x.shape)
        #print(z.shape)
        #print(self.W.shape)
        #print(y.shape)
        z = torch.einsum("ij,mnj->imn", z, self.W)
        #print(z.shape)
        embedding_sum = torch.einsum("ijk,ik->jk", z, y)

        self.m = self.hparams.gamma * self.m + (1 - self.hparams.gamma) * embedding_sum

    def forward(self, x):
        z = x
        y_pred = self.rbf(z)
        return (z, y_pred)
