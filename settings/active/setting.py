from ..base import Setting
from ..base.environment import ObservationType, ActionType, RewardType
from .environment import ActiveEnvironment as ActiveDataLoader
from dataclasses import dataclass

@dataclass
class ActiveSetting(Setting[ActiveDataLoader[ObservationType, ActionType, RewardType]]):
    """LightningDataModule for an 'active' setting.
    
    TODO: Use this for something like RL or Active Learning.
    """