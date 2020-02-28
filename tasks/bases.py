from abc import ABC, abstractmethod
from dataclasses import InitVar, dataclass, field
from typing import Any, Callable, Generic, List, Optional, Tuple, TypeVar, ClassVar, Type

import torch
from torch import Tensor, nn, optim
from torch.nn import functional as F


class AuxiliaryTask(nn.Module):
    """ Represents an additional loss to apply to a `Classifier`.

    The main logic should be implemented in the `get_loss` method.

    In general, it should apply some deterministic transformation to its input,
    and treat that same transformation as a label to predict.
    That loss should be backpropagatable through the feature extractor (the
    `encoder` attribute). 
    """
    input_shape: Tuple[int, ...] = ()
    hidden_size: int = -1
    encoder: nn.Module
    classifier: nn.Module
    preprocessing: Callable[[Tensor], Tensor]
    
    @dataclass
    class Options:
        """Settings for this Auxiliary Task. """
        # Coefficient used to scale the task loss before adding it to the total.
        coefficient: float = 0.

    def __init__(self, options: Options=None):
        """Creates a new Auxiliary Task to further train the encoder.
        
        Should use the `encoder` and `classifier` components of the parent
        `Classifier` instance.
        
        NOTE: Since this object will be stored inside the `tasks` list in the
        model, we can't pass a reference to the parent here, otherwise the
        parent would hold a reference to itself inside its `.modules()`. 
        
        Parameters
        ----------
        - encoder : nn.Module
        
            The encoder (or feature extractor) of the parent `Classifier`.
        - classifier : nn.Module
        
            The classifier (logits) layer of the parent `Classifier`.
        - options : TaskOptions, optional, by default None
        
            The `Options` related to this task, containing the loss 
            coefficient used to scale this task, as well as any other additional
            hyperparameters specific to this `AuxiliaryTask`.
        """
        super().__init__()
        self.Options.task = type(self)
        # self.encoder: nn.Module = encoder
        # self.classifier: nn.Module = classifier
        self.options: AuxiliaryTask.Options = options if options is not None else self.Options()
        # self.preprocessing: Callable[[Tensor], Tensor] = preprocessing or (lambda x: x)

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(self.preprocessing(x))

    @abstractmethod
    def get_loss(self, x: Tensor, h_x: Tensor, y_pred: Tensor, y: Tensor=None) -> Tensor:
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
    
    @property
    def coefficient(self) -> float:
        return self.options.coefficient

    @property
    def name(self) -> str:
        return type(self).__qualname__
    
    @property
    def enabled(self) -> bool:
        return self.coefficient != 0

TaskType = TypeVar("TaskType", bound=AuxiliaryTask)