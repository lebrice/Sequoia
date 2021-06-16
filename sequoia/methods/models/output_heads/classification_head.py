from dataclasses import dataclass
from typing import Dict, List, Union, Type, ClassVar, Optional

import gym
import torch
from gym import spaces
from torch import Tensor, nn, LongTensor
from simple_parsing import list_field

from sequoia.common.hparams import uniform, categorical
from sequoia.common import Batch, ClassificationMetrics, Loss
from sequoia.settings import Observations, Actions, Rewards

from .output_head import OutputHead
from ..forward_pass import ForwardPass
from ..fcnet import FCNet

# TODO: This is based on 'Actions' which is currently basically the same for all settings
# However, there should probably have a different `Action` class on a
# IncrementalSLSetting("mnist") vs IncrementalSLSetting("some_regression_dataset")!
# IDEA: What if Settings were actually meta-classes, where the 'instances' were for a
# particular choice of dataset? (e.g. `IncrementalSLSetting("mnist")` -> <type SplitMnistSetting>)
# This would maybe look a bit like the 'fully compositional' approach as well?


@dataclass(frozen=True)
class ClassificationOutput(Actions):
    """ Typed dict-like class that represents the 'forward pass'/output of a
    classification head, which correspond to the 'actions' to be sent to the
    environment, in the general formulation.
    """
    y_pred: Union[LongTensor, Tensor]
    logits: Tensor

    @property
    def action(self) -> LongTensor:
        return self.y_pred
    
    @property
    def y_pred_log_prob(self) -> Tensor:
        """ returns the log probabilities for the chosen actions/predictions. """
        return self.logits[:, self.y_pred]

    @property
    def y_pred_prob(self) -> Tensor:
        """ returns the log probabilities for the chosen actions/predictions. """
        return self.probabilities[self.y_pred]

    @property
    def probabilities(self) -> Tensor:
        """ Returns the normalized probabilies for each class, i.e. the
        softmax-ed version of `self.logits`.
        """
        return self.logits.softmax(-1)


class ClassificationHead(OutputHead):

    @dataclass
    class HParams(FCNet.HParams, OutputHead.HParams):
        """ Hyper-parameters of the OutputHead used for classification. """

        # NOTE: These hparams were basically copied over from FCNet.HParams, just so its a
        # bit more visible.

        available_activations: ClassVar[Dict[str, Type[nn.Module]]] = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "elu": nn.ELU, # No idea what these do, but hey, they are available!
            "gelu": nn.GELU,
            "relu6": nn.ReLU6,
        }
        # Number of hidden layers in the output head.
        hidden_layers: int = uniform(0, 3, default=0)
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

    def __init__(self,
                 input_space: gym.Space,
                 action_space: gym.Space,
                 reward_space: gym.Space = None,
                 hparams: "ClassificationHead.HParams" = None,
                 name: str = "classification"):
        super().__init__(
            input_space=input_space,
            action_space=action_space,
            reward_space=reward_space,
            hparams=hparams,
            name=name,
        )
        self.hparams: ClassificationHead.HParams

        assert isinstance(action_space, spaces.Discrete)
        output_size = action_space.n
        self.dense = FCNet(
            in_features=self.input_size,
            out_features=output_size,
            hparams=self.hparams,
        )
        # if output_size == 2:
        #     # TODO: Should we be using this loss instead?
        #     self.loss_fn = nn.BCEWithLogitsLoss()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, observations: Observations, representations: Tensor) -> ClassificationOutput:
        # TODO: This should probably take in a dict and return a dict, or something like that?
        # TODO: We should maybe convert this to also return a dict instead
        # of a Tensor, just to be consistent with everything else. This could
        # also maybe help with having multiple different output heads, each
        # having a different name and giving back a dictionary of their own
        # forward pass tensors (if needed) and predictions?
        logits = self.dense(representations)
        y_pred = logits.argmax(dim=-1)
        return ClassificationOutput(
            logits=logits,
            y_pred=y_pred,
        )

    def get_loss(self, forward_pass: ForwardPass, actions: ClassificationOutput, rewards: Rewards) -> Loss:
        logits: Tensor = actions.logits
        y_pred: Tensor = actions.y_pred
        rewards = rewards.to(logits.device)
        
        y: Tensor = rewards.y

        n_classes = logits.shape[-1]
        # Could remove these: just used for debugging.
        assert len(y.shape) == 1, y.shape
        assert not torch.is_floating_point(y), y.dtype
        assert 0 <= y.min(), y
        assert y.max() < n_classes, y

        loss = self.loss_fn(logits, y)
        
        assert loss.shape == ()
        metrics = ClassificationMetrics(y_pred=logits, y=y)
        
        assert self.name, "Output Heads should have a name!"
        loss_object = Loss(
            name=self.name,
            loss=loss,
            # NOTE: we're passing the tensors to the Loss object because we let
            # it create the Metrics for us automatically.
            metrics={self.name: metrics},
        )
        return loss_object
