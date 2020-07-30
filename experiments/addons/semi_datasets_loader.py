

from utils.json_utils import Serializable
from typing import Callable, ClassVar, Dict, Tuple, Type

from torch.utils.data import DataLoader, Dataset, Sampler, TensorDataset
from simple_parsing import choice, field, mutable_field, subparsers
from datasets import Datasets, DatasetConfig

@dataclass
class DatasetsLoader(Serializable):
    #labeled datasets
    datasets_labeled: choice({
            d.name: d.value for d in Datasets
        }, default=Datasets.mnist.name)
    #unlabeled datasets to combine
    datasets_unlabeled: choice({
            d.name: d.value for d in Datasets
        }, default=Datasets.mnist.name)
