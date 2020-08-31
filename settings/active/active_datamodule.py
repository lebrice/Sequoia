from .active_dataloader import ActiveDataLoader
from pytorch_lightning import LightningDataModule
from abc import ABC



class ActiveLightningDataModule(LightningDataModule, ABC):
    """ 'Active' version of a LightningDataModule.

    The `train/val/test_dataloaders()` return `ActiveDataLoader` objects instead
    of regular `DataLoader`s. 
    """
    
    def train_dataloader(self, *args, **kwargs) -> ActiveDataLoader:
        return super().train_dataloader(*args, **kwargs)