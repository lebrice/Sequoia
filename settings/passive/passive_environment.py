from torch.utils.data import DataLoader
from settings.base.environment import EnvironmentBase, ObservationType, RewardType, ActionType
from typing import *


class PassiveEnvironment(DataLoader, EnvironmentBase, Generic[ObservationType, RewardType]):
    """Environment in which actions have no influence on future observations.
    
    Normal supervised datasets such as Mnist, ImageNet, etc. fit under this
    category. 
    For now, this is exactly the same as a DataLoader, basically.

    TODO: Could instead subclass the ActiveEnvironment class and add a little
    'mechanism' to yield tuples instead of observations and rewards separately.
    """
    def __init__(self, dataset, pretend_to_be_active: bool = False, **kwargs):
        # TODO: When True, withold the labels from the yielded batches until a
        # prediction is received through in the 'send' method.
        self.pretend_to_be_active = pretend_to_be_active
        self.labels: Optional[Any] = None
        super().__init__(dataset=dataset, **kwargs)
    
    # def __next__(self) -> Tuple[ObservationType, RewardType]:
    #     """ Generate the next observation. """
    #     return super().__next__()

    def __iter__(self) -> Iterable[Tuple[ObservationType, RewardType]]:
        """ Iterate over the environment, yielding batches of Observations (x) and rewards (y) """
        for batch in super().__iter__():
            if not self.pretend_to_be_active:
                yield batch
            else:
                assert isinstance(batch, tuple), "Can only pretend to be active if the batches have labels (are tuples)!"
                samples, *self.labels = batch
                yield samples

    def send(self, action: Any) -> None:
        """ Return the withheld labels when in 'active' mode, and does nothing
        otherwise.
        """
        if self.pretend_to_be_active:
            return self.labels
        # TODO: What do to in this case? The loader is receiving an action,
        # but it already gave back the labels!
    
    def close(self):
        pass
