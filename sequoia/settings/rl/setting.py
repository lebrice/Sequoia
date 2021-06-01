from dataclasses import dataclass
from typing import Optional, Union, Sequence
from sequoia.settings.base import Setting
from sequoia.settings.base.environment import ActionType, ObservationType, RewardType
from .environment import ActiveEnvironment


@dataclass
class RLSetting(Setting[ActiveEnvironment[ObservationType, ActionType, RewardType]]):
    """LightningDataModule for an 'active' setting.
    
    This is to be the parent of settings like RL or maybe Active Learning.
    """

    @dataclass(frozen=True)
    class Observations(Setting.Observations):
        """ Observations in a continual RL Setting. """

        # Just as a reminder, these are the fields defined in the base classes:
        # x: Tensor
        # task_labels: Union[Optional[Tensor], Sequence[Optional[Tensor]]] = None

        # The 'done' part of the 'step' method. We add this here in case a
        # method were to iterate on the environments in the dataloader-style so
        # they also have access to those (i.e. for the BaselineMethod).
        done: Optional[Union[bool, Sequence[bool]]] = None
        # Same, for the 'info' portion of the result of 'step'.
        # TODO: If we add the 'task space' (with all the attributes, for instance
        # then add it to the observations using the `AddInfoToObservations`.
        # info: Optional[Sequence[Dict]] = None
        
    @dataclass(frozen=True)
    class Actions(Setting.Actions):
        pass
    
    @dataclass(frozen=True)
    class Rewards(Setting.Rewards):
        pass
