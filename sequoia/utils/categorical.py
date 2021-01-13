from torch.distributions import Categorical as Categorical_
from typing import Optional, Iterable, Union, Any
from torch import Tensor

class Categorical(Categorical_):
    """ Simple little addition to the `torch.distributions.Categorical`,
    allowing it to be 'split' into a sequence of distributions (to help with the
    splitting in the output
    heads)
    """
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
