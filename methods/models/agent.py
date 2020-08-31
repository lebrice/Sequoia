
"""Base class for an Agent interacting with an Active environments (ActiveDataLoader)

This is meant to be 'more general' than the 'Model' class, which is made for passive environments (regular dataloaders)
"""
from pytorch_lightning import LightningModule, LightningDataModule
from typing import Generic, TypeVar
from settings import RLSetting

from dataclasses import dataclass

SettingType = TypeVar("SettingType", bound=RLSetting)



class Agent(LightningModule, Generic[SettingType]):
    """ LightningModule that interacts with `ActiveDataLoader` dataloaders.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)