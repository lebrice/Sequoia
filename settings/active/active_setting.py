from dataclasses import dataclass

from .. import Setting
from ..base.environment import ActionType, ObservationType, RewardType
from .active_dataloader import ActiveDataLoader


@dataclass
class ActiveSetting(Setting[ActiveDataLoader[ObservationType, ActionType, RewardType]]):
    """LightningDataModule for an 'active' setting.
    
    TODO: Use this for something like RL or Active Learning.
    """
