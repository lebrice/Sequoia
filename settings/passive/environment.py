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
    def __next__(self) -> Tuple[ObservationType, RewardType]:
        """ Generate the next observation. """
        return super().__next__()
    
    def __iter__(self) -> Iterable[Tuple[ObservationType, RewardType]]:
        """ Iterate over the environment, yielding batches of Observations (x) and rewards (y) """
        yield from super().__iter__()
    
    def send(self, action: Any) -> None:
        """ Unused, since the environment is passive."""
        pass
    
    def close(self):
        pass
