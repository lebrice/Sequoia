from abc import ABC, abstractmethod
from dataclasses import InitVar, dataclass, field
from typing import Callable, ClassVar, Dict, Optional, Tuple, TypeVar

import torch
from pytorch_lightning import LightningModule
from torch import Tensor, nn, optim
from torch.nn import functional as F

from common.loss import Loss
from common.task import Task
from utils import cuda_available
from utils.json_utils import Serializable


class AuxiliaryTask(nn.Module):
    """ Represents an additional loss to apply to a `Classifier`.

    The main logic should be implemented in the `get_loss` method.

    In general, it should apply some deterministic transformation to its input,
    and treat that same transformation as a label to predict.
    That loss should be backpropagatable through the feature extractor (the
    `encoder` attribute). 
    """
    name: ClassVar[str] = ""
    input_shape: ClassVar[Tuple[int, ...]] = ()
    hidden_size: ClassVar[int] = -1

    _model: ClassVar[LightningModule]
    # Class variables for holding the Modules shared with with the classifier. 
    encoder: ClassVar[nn.Module]
    classifier: ClassVar[nn.Module]  # type: ignore

    preprocessing: ClassVar[Callable[[Tensor, Optional[Tensor]], Tuple[Tensor, Optional[Tensor]]]]
    
    @dataclass
    class Options(Serializable):
        """Settings for this Auxiliary Task. """
        # Coefficient used to scale the task loss before adding it to the total.
        coefficient: float = 0.

    def __init__(self, *args, options: Options = None, **kwargs):
        """Creates a new Auxiliary Task to further train the encoder.
        
        Should use the `encoder` and `classifier` components of the parent
        `Classifier` instance.
        
        NOTE: Since this object will be stored inside the `tasks` dict in the
        model, we can't pass a reference to the parent here, otherwise the
        parent would hold a reference to itself inside its `.modules()`, so
        there would be an infinite recursion problem. 
        
        Parameters
        ----------
        - options : AuxiliaryTask.Options, optional, by default None
        
            The `Options` related to this task, containing the loss 
            coefficient used to scale this task, as well as any other additional
            hyperparameters specific to this `AuxiliaryTask`.
        - name: str, optional, by default None

            The name of this auxiliary task. When not given, the name of the
            class is used.
        """
        super().__init__()
        # If we are given the coefficient as a constructor argument, for
        # instance, then we create the Options for this auxiliary task.
        self.options = options or type(self).Options(*args, **kwargs)
        self.device: torch.device = torch.device("cuda" if cuda_available else "cpu")

    def encode(self, x: Tensor) -> Tensor:
        x, _ = AuxiliaryTask.preprocessing(x, None)
        return AuxiliaryTask.encoder(x)

    def logits(self, h_x: Tensor) -> Tensor:
        return AuxiliaryTask.classifier(h_x)

    @abstractmethod
    def get_loss(self, forward_pass: Dict[str, Tensor], y: Tensor=None) -> Loss:
        """Calculates the Auxiliary loss for the input `x`.
        
        The parameters `h_x`, `y_pred` are given for convenience, so we don't
        re-calculate the forward pass multiple times on the same input.
        
        Parameters
        ----------
        - forward_pass: Dict[str, Tensor] containing:
            - 'x' : Tensor
            
                The input samples.
            - 'h_x' : Tensor

                The hidden vector, or hidden features, which corresponds to the
                output of the feature extractor (should be equivalent to 
                `self.encoder(x)`). Given for convenience, when available.

            - 'y_pred' : Tensor
        
                The predicted labels. 
        - y : Tensor, optional, by default None
        
            The true labels for each sample. Note that this is the label of the
            output head's task, not of an auxiliary task.
        
        Returns
        -------
        Tensor
            The loss, not scaled.
        """
        pass

    @property
    def coefficient(self) -> float:
        return self.options.coefficient

    @coefficient.setter
    def coefficient(self, value: float) -> None:
        if self.enabled and value == 0:
            self.disable()
        elif self.disabled and value != 0:
            self.enable()
        self.options.coefficient = value

    def enable(self) -> None: 
        """ Enable this auxiliary task. 
        This could be used to create/allocate resources to this task.
        """
        pass

    def disable(self) -> None:
        """ Disable this auxiliary task and sets its coefficient to 0. 
        This could be used to delete/deallocate resources used by this task.
        """
        self.options.coefficient = 0.

    @property
    def enabled(self) -> bool:
        return self.coefficient != 0
    
    @property
    def disabled(self) -> bool:
        return not self.enabled

    def on_task_switch(self, task_id: int) -> None:
        """ Executed when the task switches (to either a new or known task). """

    @property
    def model(self) -> LightningModule:
        return type(self)._model
