""" Abstract base class for an output head of the BaselineModel. """
import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Type, Callable, Sequence
import gym
import numpy as np
from gym import spaces
from gym.spaces.utils import flatdim
from simple_parsing import list_field, choice, mutable_field
from torch import Tensor, nn
from torch.nn import Flatten  # type: ignore
from torch.optim.optimizer import Optimizer

from sequoia.common.loss import Loss
from sequoia.common.metrics import ClassificationMetrics, get_metrics
from sequoia.settings import Actions, Observations, Rewards, Setting
from sequoia.utils import Parseable, get_logger
from sequoia.utils.serialization import Serializable
from sequoia.utils.utils import camel_case, remove_suffix

from ..forward_pass import ForwardPass
logger = get_logger(__file__)

@dataclass
class DenseHParams(Serializable, Parseable):
    available_activations: ClassVar[Dict[str, Type[nn.Module]]] = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
    }

    # Number of hidden layers in the output head.
    hidden_layers: int = 1
    # Number of neurons in each hidden layer of the output head.
    # If a single value is given, than each of the `hidden_layers` layers
    # will have that number of neurons. 
    # If `n > 1` values are given, then `hidden_layers` must either be 0 or
    # `n`, otherwise a RuntimeError will be raised.
    hidden_neurons: List[int] = list_field(64)
    activation: Type[nn.Module] = choice(available_activations, default="tanh")

    def __post_init__(self):
        # no value passed to --hidden_layers
        if isinstance(self.activation, str):
            self.activation = self.available_activations[self.activation.lower()]
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


class OutputHead(nn.Module, ABC):
    """Module for the output head of the model.
    
    This output head is meant for classification, but you could inherit from it
    and customize it for doing something different like RL or reconstruction, 
    for instance.
    """
    # TODO: Rename this to 'output' and create some ClassificationHead,
    # RegressionHead, ValueHead, etc. subclasses with the corresponding names.
    name: ClassVar[str] = "classification"
    
    # Reference to the optimizer of the BaselineModel.
    base_model_optimizer: ClassVar[Optimizer]

    @dataclass
    class HParams(DenseHParams):
        """ Hyperparameters of the output head. """
        
    def __init__(self,
                 input_space: gym.Space,
                 action_space: gym.Space,
                 reward_space: gym.Space = None,
                 hparams: "OutputHead.HParams" = None,
                 name: str = ""):
        super().__init__()
        
        self.input_space = input_space
        self.action_space = action_space
        self.reward_space = reward_space or spaces.Box(-np.inf, np.inf, ())
        self.input_size = flatdim(input_space)
        self.hparams = hparams or self.HParams()
        if not isinstance(self.hparams, self.HParams):
            # Upgrade the hparams to the right type, if needed.
            self.hparams = self.upgrade_hparams()
        self.name = name or type(self).name

    def make_dense_network(self,
                           in_features: int,
                           hidden_neurons: Sequence[int],
                           out_features: int,
                           activation: Type[nn.Module]=nn.ReLU):
        hidden_layers: List[nn.Module] = []
        output_size = out_features
        for i, neurons in enumerate(hidden_neurons):
            out_features = neurons
            hidden_layers.append(nn.Linear(in_features, out_features))
            hidden_layers.append(activation())
            in_features = out_features # next input size is output size of prev.

        return nn.Sequential(
            Flatten(),
            *hidden_layers,
            nn.Linear(in_features, output_size)
        )

    @abstractmethod
    def forward(self, observations: Setting.Observations, representations: Tensor) -> Setting.Actions:
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
    def get_loss(self, forward_pass: ForwardPass, actions: Actions, rewards: Rewards) -> Loss:
        """ Given the forward pass,(a dict-like object that includes the
        observations, representations and actions, the actions produced by this
        output head and the resulting rewards, returns a Loss to use.
        """

    def upgrade_hparams(self):
        """Upgrades the hparams at `self.hparams` to the right type for this
        output head (`type(self).HParams`), filling in any missing values by
        parsing them from the command-line.

        Returns
        -------
        type(self).HParams
            Hparams of the type `self.HParams`, with the original values
            preserved and any new values parsed from the command-line.
        """
        # NOTE: This (getting the wrong hparams class) could happen for
        # instance when parsing a BaselineMethod from the command-line, the
        # default type of hparams on the method is BaselineModel.HParams,
        # whose `output_head` field doesn't have the right type exactly.
        current_hparams = self.hparams.to_dict()
        # TODO: If a value is not at its current default, keep it.
        default_hparams = self.HParams()
        missing_fields = [
            f.name for f in dataclasses.fields(self.HParams)
            if f.name not in current_hparams
            or current_hparams[f.name] == getattr(type(self.hparams)(), f.name, None)
            or current_hparams[f.name] == getattr(default_hparams, f.name)
        ]
        logger.warning(RuntimeWarning(
            f"Upgrading the hparams from type {type(self.hparams)} to "
            f"type {self.HParams}. This will try to fetch the values for "
            f"the missing fields {missing_fields} from the command-line. "
        ))
        # Get the missing values

        if self.hparams._argv:
            return self.HParams.from_args(argv=self.hparams._argv, strict=False)
        hparams = self.HParams.from_args(argv=self.hparams._argv, strict=False)
        for missing_field in missing_fields:
            current_hparams[missing_field] = getattr(hparams, missing_field)
        return self.HParams(**current_hparams)  
