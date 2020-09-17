import math
from typing import Iterable, List, Sequence, Tuple, TypeVar

import torch
from torch.utils.data import IterableDataset
from utils.logging_utils import get_logger

logger = get_logger(__file__)
T = TypeVar("T")

class ZipDataset(IterableDataset):
    def __init__(self, datasets: Sequence[Sequence[T]]):
        self.all_datasets = list(datasets)

    @property
    def datasets(self) -> List[Sequence[T]]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            logger.debug(f"Single-process data loading.")
            return self.all_datasets
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil(len(self.all_datasets) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.all_datasets))
            logger.debug(f"Dataloader worker {worker_info.id} datasets {list(range(iter_start, iter_end))}")
        datasets = self.all_datasets[iter_start:iter_end]
        return datasets

    def __iter__(self) -> Iterable[Tuple[T, ...]]:
        for batches in zip(*self.datasets):
            yield list(batches)

    def __next__(self) -> List[T]:
        logger.debug(f"Inside __next__()")
        return [next(d) for d in self.datasets]

    def __getitem__(self, index: int) -> List[T]:
        logger.debug(f"Inside __getitem__")
        return [d[index] for d in self.datasets]

    def send(self, actions: Sequence) -> List:
        logger.debug(f"Inside __send__")
        return [d.send(action) for d, action in zip(self.datasets, actions)]


