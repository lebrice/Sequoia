from typing import *

from common.transforms import Compose
from torch.utils.data import DataLoader, Dataset, IterableDataset

from ..base.environment import (Actions, ActionType, Environment, Observations,
                                ObservationType, Rewards, RewardType)


class PassiveEnvironment(DataLoader, Environment[Union[ActionType,
                                                       Tuple[ObservationType,
                                                             ActionType]],
                                                 ActionType,
                                                 RewardType]):
    """Environment in which actions have no influence on future observations.
    
    Can either be iterated on like a normal dataset, in which case it gives back
    the observation and the reward at the same time, or as an Active dataset,
    where it gives the reward only after an action (doesn't matter what action)
    is sent.
    
    Normal supervised datasets such as Mnist, ImageNet, etc. fit under this
    category. Similarly to Environment, this just adds some methods on top of
    the usual PyTorch DataLoader.
    """
    def __init__(self,
                 dataset: Union[IterableDataset, Dataset],
                 pretend_to_be_active: bool = False,
                 batch_transforms: List[Callable] = None,
                 **kwargs):
        """Creates the DataLoader/Environment for the given dataset.

        Parameters
        ----------
        dataset : [type]
            The dataset to iterate on.

        pretend_to_be_active : bool, optional
            Wether to withhold , by default False
        
        batch_transforms : List[Callable], optional
            [description], by default None
        
        **kwargs:
            The rest of the usual dataloader kwargs.
        """
        # TODO: When True, withold the labels from the yielded batches until a
        # prediction is received through in the 'send' method.
        self.pretend_to_be_active = pretend_to_be_active
        self.labels: Optional[Any] = None
        self.batch_transforms: Compose = Compose(batch_transforms or [])
        super().__init__(dataset=dataset, **kwargs)
    
    # def __next__(self) -> Tuple[ObservationType, RewardType]:
    #     """ Generate the next observation. """
    #     return super().__next__()

    def __iter__(self) -> Iterable[Tuple[ObservationType, RewardType]]:
        """ Iterate over the environment, yielding batches of Observations (x) and rewards (y) """
        for batch in super().__iter__():
            batch = self.batch_transforms(batch)
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
