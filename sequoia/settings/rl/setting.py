from dataclasses import dataclass
from typing import Optional, Union, Sequence, ClassVar, Type
from sequoia.settings.base import Setting
from sequoia.settings.base.environment import ActionType, ObservationType, RewardType
from .environment import ActiveEnvironment, RLEnvironment

from .objects import Observations, ObservationType, Actions, ActionType, Rewards, RewardType


@dataclass
class RLSetting(Setting[RLEnvironment[ObservationType, ActionType, RewardType]]):
    """LightningDataModule for an 'active' setting.
    
    This is to be the parent of settings like RL or maybe Active Learning.
    """
    Observations: ClassVar[Type[ObservationType]] = Observations 
    Actions: ClassVar[Type[ActionType]] = Actions 
    Rewards: ClassVar[Type[RewardType]] = Rewards 
