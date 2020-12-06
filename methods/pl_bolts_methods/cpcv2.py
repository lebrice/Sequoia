""" TODO: Add the corresponding model from pl_bolts as a Methods targetting the
IID Setting.

Trying to follow this: https://pytorch-lightning-bolts.readthedocs.io/en/latest/self_supervised_models.html
"""
from dataclasses import dataclass

import pytorch_lightning as pl
import torch

from pl_bolts.datamodules import (CIFAR10DataModule, ImagenetDataModule,
                                  MNISTDataModule, STL10DataModule)
from pl_bolts.models import AE, GPT2
from pl_bolts.models.self_supervised import CPCV2
from pl_bolts.models.self_supervised.cpc import (CPCEvalTransformsCIFAR10,
                                                 CPCTrainTransformsCIFAR10)
# from pl_bolts.models.vision.image_gpt import ImageGPT
# from pl_bolts.models.gans import GAN

from settings import IIDSetting, IIDResults
from methods import Method


# datamodule = CIFAR10DataModule(
#     "data",
#     num_workers=24,
#     batch_size=32,
# )
# datamodule.train_transforms = CPCTrainTransformsCIFAR10()
# datamodule.val_transforms = CPCEvalTransformsCIFAR10()

# # # load resnet18 pretrained using CPC on imagenet
# model = CPCV2(pretrained='resnet18', datamodule=datamodule)
# # cpc_resnet18 = model.encoder
# # cpc_resnet18.freeze()

# trainer = pl.Trainer(gpus=1, fast_dev_run=True)
# trainer.fit(model)
# test_results = trainer.test(model)
# print(f"Test outputs: {test_results}")
# exit()
# it supports any torchvision resnet
# model = CPCV2(pretrained='resnet50')


# seq_len = 17
# batch_size = 32
# vocab_size = 16
# x = torch.randint(0, vocab_size, (seq_len, batch_size))
# model = GPT2(embed_dim=32, heads=2, layers=2, num_positions=seq_len, vocab_size=vocab_size, num_classes=4)
# results = model(x)
# print(results)
# exit()

# dm = MNISTDataModule('data')
# model = ImageGPT(dm)


@dataclass
class CPCV2Method(Method[IIDSetting], target_setting=IIDSetting):
    """ Adds existing methods from Pytorch Lightnign Bolts to use in an IID Setting.
    """
    # HyperParameters of the method.
    # hparams: HParams = mutable_field(HParams)
    # # Options for the Trainer object.
    # trainer_options: CLTrainerOptions = mutable_field(CLTrainerOptions)
    # # Configuration options for the experimental setup (log_dir, cuda, etc).
    # config: Config = mutable_field(Config)

    def apply_to(self, setting: IIDSetting) -> IIDResults:
        """ Applies this method to the particular experimental setting.
        
        Extend this class and overwrite this method to customize training.      
        """
        if not self.is_applicable(setting):
            raise RuntimeError(
                f"Can only apply methods of type {type(self)} on settings "
                f"that inherit from {type(self).target_setting}. "
                f"(Given setting is of type {type(setting)})."
            )

        # Seed everything first:
        self.config.seed_everything()
        setting.configure(config=self.config)
        from common.transforms import ToTensor
        # setting.transforms = [ToTensor(), CPCTrainTransformsCIFAR10()]
        setting.train_transforms = [ToTensor(), CPCTrainTransformsCIFAR10()]
        setting.val_transforms = [ToTensor(), CPCEvalTransformsCIFAR10()]

        # TODO: Seems a weird that we would have to do this.
        setting.data_dir = self.config.data_dir
        setting.config = self.config
        setting.batch_size = 16
        
        # # load resnet18 pretrained using CPC on imagenet
        model = CPCV2(pretrained='resnet18', datamodule=setting)
        # cpc_resnet18 = model.encoder
        # cpc_resnet18.freeze()

        
        trainer = pl.Trainer(gpus=1, fast_dev_run=True)
        trainer.fit(model, datamodule=setting)
        test_results = trainer.test(model)
        print(f"Test outputs: {test_results}")
        raise NotImplementedError("TODO: The CPCV2 model doesn't have a 'test_step' method.")


if __name__ == "__main__":
    PLBoltsMethod.main()
