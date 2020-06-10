from torchvision.datasets import VisionDataset
from typing import Tuple
import numpy as np
from .subset import Subset
import logging
logger = logging.getLogger(__file__)


def train_valid_split(train_dataset: VisionDataset, valid_fraction: float=0.2) -> Tuple[VisionDataset, VisionDataset]:
    n = len(train_dataset)
    valid_len: int = int((n * valid_fraction))
    train_len: int = n - valid_len
    
    indices = np.arange(n, dtype=int)
    np.random.shuffle(indices)
    
    valid_indices = indices[:valid_len]
    train_indices = indices[valid_len:]

    train = Subset(train_dataset, train_indices)
    valid = Subset(train_dataset, valid_indices)
    logger.info(f"Training samples: {len(train)}, Valid samples: {len(valid)}")
    return train, valid
