import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from socket import gethostname
from typing import Callable, ClassVar, Dict, Tuple, Type

from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision.datasets import ImageNet, VisionDataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from simple_parsing import field
from .dataset_config import DatasetConfig


def get_imagenet_location() -> Path:
    from socket import gethostname
    hostname = gethostname()
    # For each hostname prefix, the location where the torchvision ImageNet dataset can be found.
    # TODO: Add the location for your own machine.
    imagenet_locations: Dict[str, Path] = {
        "mila": Path("/network/datasets/imagenet.var/imagenet_torchvision"),
        "": Path("/network/datasets/imagenet.var/imagenet_torchvision"),
    }
    for prefix, v in imagenet_locations.items():
        if hostname.startswith(prefix):
            return v
    if "IMAGENET_DIR" in os.environ:
        return Path(os.environ["IMAGENET_DIR"])
    raise RuntimeError(
        f"Could not find the ImageNet dataset on this machine with hostname "
        f"{hostname}. Known <prefix --> location> pairs: {imagenet_locations}"
    )

<<<<<<< HEAD
@dataclass(frozen=True)
class ImageNetConfig(DatasetConfig):
    dataset_class: Type[VisionDataset] = ImageNet
=======
@dataclass
class ImageNetConfig(DatasetConfig):
    dataset_class: Type[VisionDataset] = field(default=ImageNet, encoding_fn=str) 
>>>>>>> master
    x_shape: Tuple[int, int, int] = (3, 224, 224)
    num_classes: int = 1000
    transforms: Callable = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace=True),
    ])
    keep_in_memory: bool=False

    def load(self, data_dir: Path=None, download: bool=False) -> Tuple[Dataset, Dataset]:
        """ Downloads the ImageNet dataset.
        """
        if data_dir is not None:
            warnings.warn(UserWarning(
                f"Using data_dir has no effect when using the ImageNet dataset."
            ))
        data_dir = get_imagenet_location()
        train = self.dataset_class(data_dir, split="train", transform=self.transforms)
        test  = self.dataset_class(data_dir, split="val", transform=self.transforms)
        return train, test
