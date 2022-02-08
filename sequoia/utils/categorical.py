from typing import Any, Iterable, Optional, Union

import torch
from torch import Tensor
from torch.distributions import Categorical as Categorical_


class Categorical(Categorical_):
    """Simple little addition to the `torch.distributions.Categorical`,
    allowing it to be 'split' into a sequence of distributions (to help with the
    splitting in the output
    heads)
    """

    def __init__(
        self,
        probs: Optional[Tensor] = None,
        logits: Optional[Tensor] = None,
        validate_args: bool = None,
    ):
        super().__init__(probs=probs, logits=logits, validate_args=validate_args)
        self._device: torch.device = probs.device if probs is not None else logits.device

    def __getitem__(self, index: Optional[int]) -> "Categorical":
        return Categorical(logits=self.logits[index])
        # return Categorical(probs=self.probs[index])

    def __iter__(self) -> Iterable["Categorical"]:
        for index in range(self.logits.shape[0]):
            yield self[index]

    def __add__(self, other: Union["Categorical_", Any]) -> "Categorical":
        # Idea:, how about we return a wrapped version of `self` whose
        # 'sample' returns self.sample() + `other`?
        return NotImplemented

    def __mul__(self, other: Union["Categorical_", Any]) -> "Categorical":
        # Idea: Idea, how about we return a wrapped version of `self` whose
        # 'sample' returns self.sample() * `other`?
        return NotImplemented

    @property
    def device(self) -> torch.device:
        """The device of the tensors of this distribution.

        @lebrice: Not sure why this isn't already part of torch.Distribution base-class.
        """
        return self._device

    def to(self, device: Union[str, torch.device]) -> "Categorical":
        """Moves this distribution to another device.

        @lebrice: Not sure why this isn't already part of torch.Distribution base-class.
        """
        return type(self)(logits=self.logits.to(device=device))
