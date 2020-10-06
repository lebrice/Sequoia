from typing import *

from common.transforms import Compose
from torch.utils.data import DataLoader, Dataset, IterableDataset

from ..base.environment import (Actions, ActionType, Environment, Observations,
                                ObservationType, Rewards, RewardType)


class PassiveEnvironment(DataLoader, Environment[Tuple[ObservationType,
                                                       Optional[ActionType]],
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
                 observations_type: Type[ObservationType],
                 rewards_type: Type[RewardType],
                 actions_type: Type[ActionType],
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
        
        batch_transforms : List[Callable[Tuple[Tensor, ...]]], optional
            Transforms that operate on all the entire batch of tensors that is
            returned by the dataset, by default None.
        
        **kwargs:
            The rest of the usual dataloader kwargs.
        """
        # TODO: When True, withold the labels from the yielded batches until a
        # prediction is received through in the 'send' method.
        self.pretend_to_be_active = pretend_to_be_active
        self.observations_type = observations_type
        self.actions_type = actions_type
        self.rewards_type = rewards_type

        self.batch_transforms: Compose = Compose(batch_transforms or [])
        from common.transforms import SplitBatch
        if not any(isinstance(t, SplitBatch) for t in self.batch_transforms):
            if observations_type and rewards_type:
                self.batch_transforms.append(SplitBatch(observations_type, rewards_type))
            else:
                raise RuntimeError(
                    f"`batch_transforms` needs to contain a SplitBatch "
                    f"transform! Or, you can pass an `observations_type` and a"
                    f"`rewards_type` to the {__class__} constructor and it "
                    f"will create one for you. \n"
                    f"(transforms: {self.batch_transforms})"
                )
        
        super().__init__(dataset=dataset, **kwargs)
        self.observations: Union[Observations, Any] = None
        self.rewards: Union[Rewards, Any] = None
        
        # TODO: Create Observation / Action / Reward gym spaces!
        # IDEA: Could make use of some of the properties of the `Batch` object,
        # such as the `shapes` property.

    def __iter__(self) -> Iterable[Tuple[ObservationType, Optional[RewardType]]]:
        """Iterate over the dataset, yielding batches of Observations and
        Rewards (or None if in 'fake active' mode).
        """
        for batch in super().__iter__():
            batch = self.batch_transforms(batch)
            
            # For now, just to simplify, we assume that the batch has already
            # been split into Observations and Actions by a SplitBatch transform.
            assert len(batch) == 2
            assert isinstance(batch[0], Observations)
            assert isinstance(batch[1], Rewards)
            self.observations, self.rewards = batch
            
            if self.pretend_to_be_active:
                # TODO: Should we yield one item, or two?
                yield self.observations, None
            else:
                yield self.observations, self.rewards

    def send(self, action: Actions) -> Rewards:
        """ Return the last latch of rewards from the dataset (which were
        withheld if in 'active' mode)
        """
        assert isinstance(action, self.actions_type)
        return self.rewards
    
    def close(self):
        pass
