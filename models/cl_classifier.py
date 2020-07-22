from dataclasses import dataclass
from typing import List

from torch.utils.data import DataLoader

from config import Config
from models.classifier import Classifier
from setups.cl.base import CLSetting


class ContinualClassifier(Classifier[CLSetting]):
    """ Extension of the Classifier LightningModule aimed at CL settings.
    TODO: Add the stuff related to multihead/continual learning here?
    """

    @dataclass
    class HParams(Classifier.HParams):
        """ Hyperparameters specific to a Continual Learning classifier.
        TODO: Add any hyperparameters specific to CL here.
        """
        pass


    def __init__(self, setting: CLSetting, hparams: HParams, config: Config):
        if not isinstance(setting, CLSetting):
            raise RuntimeError(
                f"Can only apply this model on a CLSetting or "
                f"on a setting which inherits from it! "
                f"(given setting is of type {type(setting)})."
            )
        super().__init__(setting, hparams, config)

    def train_dataloaders(self, **kwargs) -> List[DataLoader]:
        """ Returns the dataloaders for all train tasks.
        See the `ClassIncrementalSetting` class for more info.
        """
        kwargs = self.dataloader_kwargs(**kwargs)
        return self.setting.train_dataloaders(**kwargs)
    
    def val_dataloaders(self, **kwargs) -> List[DataLoader]:
        """ Returns the dataloaders for all validation tasks.
        See the `ClassIncrementalSetting` class for more info.
        """
        kwargs = self.dataloader_kwargs(**kwargs)
        return self.setting.val_dataloaders(**kwargs)
    
    def test_dataloaders(self, **kwargs) -> List[DataLoader]:
        """ Returns the dataloaders for all test tasks.
        See the `ClassIncrementalSetting` class for more info.
        """
        kwargs = self.dataloader_kwargs(**kwargs)
        return self.setting.test_dataloaders(**kwargs)
