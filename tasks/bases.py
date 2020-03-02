from abc import ABC, abstractmethod
from dataclasses import InitVar, dataclass, field
from typing import *
from typing import ClassVar

import torch
from torch import Tensor, nn, optim
from torch.nn import functional as F
from common.losses import LossInfo

class AuxiliaryTask(nn.Module):
    """ Represents an additional loss to apply to a `Classifier`.

    The main logic should be implemented in the `get_loss` method.

    In general, it should apply some deterministic transformation to its input,
    and treat that same transformation as a label to predict.
    That loss should be backpropagatable through the feature extractor (the
    `encoder` attribute). 
    """
    input_shape: ClassVar[Tuple[int, ...]] = ()
    hidden_size: ClassVar[int] = -1
    
    # Class variables for holding the Modules shared with with the classifier. 
    encoder: ClassVar[nn.Module]
    classifier: ClassVar[nn.Module]
    preprocessing: ClassVar[Callable[[Tensor], Tensor]]
    
    @dataclass
    class Options:
        """Settings for this Auxiliary Task. """
        # Coefficient used to scale the task loss before adding it to the total.
        coefficient: float = 0.

    def __init__(self, options: Options=None, *args, **kwargs):
        """Creates a new Auxiliary Task to further train the encoder.
        
        Should use the `encoder` and `classifier` components of the parent
        `Classifier` instance.
        
        NOTE: Since this object will be stored inside the `tasks` list in the
        model, we can't pass a reference to the parent here, otherwise the
        parent would hold a reference to itself inside its `.modules()`. 
        
        Parameters
        ----------
        - coefficient : float, optional, by default None

            The coefficient of the loss for this auxiliary task. When None, the
            coefficient from the `options` is used. When `options` and
            `coefficient` are both given, `coefficient` is used instead of
            `options.coefficient`.
        - options : TaskOptions, optional, by default None
        
            The `Options` related to this task, containing the loss 
            coefficient used to scale this task, as well as any other additional
            hyperparameters specific to this `AuxiliaryTask`.
        """
        super().__init__()
        self.options = options or self.Options(*args, **kwargs)
        self._coefficient = nn.Parameter(torch.Tensor([self.options.coefficient]))  # type: ignore

    def encode(self, x: Tensor) -> Tensor:
        x = AuxiliaryTask.preprocessing(x)
        return AuxiliaryTask.encoder(x)

    def logits(self, h_x: Tensor) -> Tensor:
        return AuxiliaryTask.classifier(h_x)

    @abstractmethod
    def get_loss(self, x: Tensor, h_x: Tensor, y_pred: Tensor, y: Tensor=None) -> LossInfo:
        """Calculates the Auxiliary loss for the input `x`.ABC
        
        The parameters `h_x`, `y_pred` are given for convenience, so we don't
        re-calculate the forward pass multiple times on the same input.
        
        Parameters
        ----------
        - x : Tensor
        
            The input samples.ABC
        - h_x : Tensor
        
            The hidden vector, or hidden features, which corresponds to the
            output of the feature extractor (should be equivalent to 
            `self.encoder(x)`). Given for convenience, when available.ABC
        - y_pred : Tensor
        
            The predicted (raw/unscaled) scores for each class, which 
            corresponds to the output of the classifier layer of the parent
            Model. (should be equivalent to `self.classifier(self.encoder(x))`). 
        - y : Tensor, optional, by default None
        
            The true labels for each sample. Will generally be None, as we don't
            generally use the label for Auxiliary Tasks.
            TODO: Is there any case where we might use the labels here?
        
        Returns
        -------
        Tensor
            The loss, not scaled.
        """
        pass

    def get_scaled_loss(self, x: Tensor, h_x: Tensor, y_pred: Tensor, y: Tensor=None) -> LossInfo:
        if not self.enabled:
            return LossInfo()
        loss_info = self.get_loss(x, h_x, y_pred, y)
        loss_info.add_prefix(self.name)
        if loss_info.total_loss != 0 and not loss_info.losses:
            loss_info.losses[self.name] = loss_info.total_loss

        loss_info.scale_by(self.coefficient)
        return loss_info


    @property
    def coefficient(self) -> float:
        return self._coefficient

    @coefficient.setter
    def coefficient(self, value: float) -> None:
        self._coefficient.data = value
        self.options.coefficient = value

    @property
    def name(self) -> str:
        return type(self).__qualname__
    
    @property
    def enabled(self) -> bool:
        return self.coefficient != 0

TaskType = TypeVar("TaskType", bound=AuxiliaryTask)