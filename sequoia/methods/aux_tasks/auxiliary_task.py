import typing
from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, ClassVar, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import Tensor, nn

from sequoia.common.hparams import HyperParameters, uniform
from sequoia.common.loss import Loss

if typing.TYPE_CHECKING:
    from sequoia.methods.models.base_model import Model


class AuxiliaryTask(nn.Module):
    """Represents an additional loss to apply to a `Classifier`.

    The main logic should be implemented in the `get_loss` method.

    In general, it should apply some deterministic transformation to its input,
    and treat that same transformation as a label to predict.
    That loss should be backpropagatable through the feature extractor (the
    `encoder` attribute).
    """

    name: ClassVar[str] = ""
    input_shape: ClassVar[Tuple[int, ...]] = ()
    hidden_size: ClassVar[int] = -1

    _model: ClassVar["Model"]
    # Class variables for holding the Modules shared with the classifier.
    encoder: ClassVar[nn.Module]
    output_head: ClassVar[nn.Module]  # type: ignore

    preprocessing: ClassVar[Callable[[Tensor, Optional[Tensor]], Tuple[Tensor, Optional[Tensor]]]]

    @dataclass
    class Options(HyperParameters):
        """Settings for this Auxiliary Task."""

        # Coefficient used to scale the task loss before adding it to the total.
        coefficient: float = uniform(0.0, 1.0, default=1.0)

    def __init__(self, *args, options: Options = None, name: str = None, **kwargs):
        """Creates a new Auxiliary Task to further train the encoder.

        Can use the `encoder` and `classifier` components of the parent
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
        self.name = name or type(self).name
        self.options = options or type(self).Options(*args, **kwargs)
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._disabled = False

    def encode(self, x: Tensor) -> Tensor:
        # x, _ = AuxiliaryTask.preprocessing(x, None)
        return AuxiliaryTask.encoder(x)

    def logits(self, h_x: Tensor) -> Tensor:
        return AuxiliaryTask.output_head(h_x)

    @abstractmethod
    def get_loss(self, forward_pass: Dict[str, Tensor], y: Tensor = None) -> Loss:
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
        """Enable this auxiliary task.
        This could be used to create/allocate resources to this task.

        NOTE: The task will not work, even after being enabled, if its
        coefficient is set to 0!
        """
        self._disabled = False

    def disable(self) -> None:
        """Disable this auxiliary task and sets its coefficient to 0.
        This could be used to delete/deallocate resources used by this task.
        """
        self._disabled = True

    @property
    def enabled(self) -> bool:
        return not self._disabled

    @property
    def disabled(self) -> bool:
        return self._disabled or self.coefficient == 0.0

    def on_task_switch(self, task_id: Optional[int]) -> None:
        """Executed when the task switches (to either a new or known task)."""

    @property
    def model(self) -> LightningModule:
        return type(self)._model

    @staticmethod
    def set_model(model: "Model") -> None:
        AuxiliaryTask._model = model

    def shared_modules(self) -> Dict[str, nn.Module]:
        """Returns any trainable modules if `self` that are shared across tasks.

        By giving this information, these weights can then be used in
        regularization-based auxiliary tasks like EWC, for example.

        By default, for auxiliary tasks, this returns nothing, for instance.
        For the base model, this returns a dictionary with the encoder, for example.
        When using only one output head (i.e. when `self.hp.multihead` is `False`), then
        this dict also includes the output head.

        Returns
        -------
        Dict[str, nn.Module]:
            Dictionary mapping from name to the shared modules, if any.
        """
        return {}
