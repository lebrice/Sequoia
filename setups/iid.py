import pl_bolts
from pl_bolts.datamodules.mnist_datamodule import MNISTDataModule
from pathlib import Path

data_dir = Path("data")
data_module = MNISTDataModule(data_dir, val_split=5000, num_workers=16, normalize=False)

exit()