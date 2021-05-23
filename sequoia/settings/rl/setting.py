from dataclasses import dataclass

from sequoia.settings.base import Setting
from sequoia.settings.base.environment import ActionType, ObservationType, RewardType
from .environment import ActiveEnvironment


@dataclass
class RLSetting(Setting[ActiveEnvironment[ObservationType, ActionType, RewardType]]):
    """LightningDataModule for an 'active' setting.
    
    This is to be the parent of settings like RL or maybe Active Learning.
    """