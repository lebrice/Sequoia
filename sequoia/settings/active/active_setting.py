from dataclasses import dataclass

from .. import Setting
from ..base.environment import ActionType, ObservationType, RewardType
from .active_dataloader import ActiveDataLoader


@dataclass
class ActiveSetting(Setting[ActiveDataLoader[ObservationType, ActionType, RewardType]]):
    """LightningDataModule for an 'active' setting.
    
    This is to be the parent of settings like RL or maybe Active Learning.
    """