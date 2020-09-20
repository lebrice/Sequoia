# from pl_bolts.models.self_supervised.simclr import SimCLRTrainDataTransform, SimCLREvalDataTransform
# from pl_bolts.models.self_supervised import SimCLR
import pytorch_lightning as pl
from pl_bolts.datamodules import (CIFAR10DataModule, FashionMNISTDataModule,
                                  MNISTDataModule, TinyCIFAR10DataModule)
from pl_bolts.models.autoencoders import AE
from pytorch_lightning import Trainer
import pytest

from .cl_trainer import CLTrainer

@pytest.mark.skip(reason="Skipping, need to figure out what the 'usual' return value of `trainer.test()` is. ")
def test_pytorch_lightning_trainer_test_returns_something(tmpdir):
    dm = CIFAR10DataModule(batch_size=4)
    model = AE(input_height=dm.size()[-1])
    trainer = CLTrainer(
        fast_dev_run=True,
        default_root_dir=tmpdir,
        max_epochs=1,
        gpus=None
    )

    result = trainer.fit(model, datamodule=dm)
    assert result == 1
    results = trainer.test(verbose=True)
    print(f"results (no arguments): {results}")
    assert results == 1

    results = trainer.test(datamodule=dm, verbose=True)
    assert isinstance(results, dict)
    print(f"results (same datamodule): {results}")
    results = trainer.test(datamodule=CIFAR10DataModule("data"))
    assert isinstance(results, dict)
    print(f"results (same datamodule): {results}")
    
    results = trainer.test(datamodule=TinyCIFAR10DataModule("data"))
    print(f"Results (other datamodule): {results}")
    assert isinstance(results, dict)
