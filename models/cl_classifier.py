from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

from torch import Tensor
from torch.utils.data import DataLoader

from config import Config
from models.classifier import Classifier
from setups.cl.continual_setting import ContinualSetting


class ContinualClassifier(Classifier[ContinualSetting]):
    """ Extension of the Classifier LightningModule aimed at CL settings.
    TODO: Add the stuff related to multihead/continual learning here?
    """

    @dataclass
    class HParams(Classifier.HParams):
        """ Hyperparameters specific to a Continual Learning classifier.
        TODO: Add any hyperparameters specific to CL here.
        """
        pass


    def __init__(self, setting: ContinualSetting, hparams: HParams, config: Config):
        if not isinstance(setting, ContinualSetting):
            raise RuntimeError(
                f"Can only apply this model on a CLSetting or "
                f"on a setting which inherits from it! "
                f"(given setting is of type {type(setting)})."
            )
        super().__init__(setting, hparams, config)
        self.setting: ContinualSetting

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
    
    def _shared_step(self, batch: Tuple[Tensor, Optional[Tensor]],
                           batch_idx: int,
                           dataloader_idx: int=None,
                           loss_name: str="",
                           training: bool=True,
                    ) -> Dict:
        if dataloader_idx is not None:
            self.on_task_switch(dataloader_idx, training=training)
        return super()._shared_step(
            batch=batch,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
            loss_name=loss_name,
        )


    def on_task_switch(self, task_id: int, training: bool=False) -> None:
        """Called when switching between tasks.
        TODO: Should use this like we were doing before maybe.

        Args:
            task_id (int): the Id of the task.
            training (bool): Wether we are currently training or valid/testing.
        """
        self.setting.current_task_id = task_id
