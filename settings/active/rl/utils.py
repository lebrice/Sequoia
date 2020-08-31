from typing import Iterable, List, Sequence, Tuple, TypeVar

from torch.utils.data import IterableDataset
T = TypeVar("T")

class ZipDataset(IterableDataset):
    def __init__(self, datasets: Sequence[Sequence[T]]):
        self.datasets = list(datasets)
    
    def __iter__(self) -> Iterable[Tuple[T, ...]]:
        for batches in zip(*self.datasets):
            yield list(batches)

    def __next__(self) -> List[T]:
        return [next(d) for d in self.datasets]

    def __getitem__(self, index: int) -> List[T]:
        return [d[index] for d in self.datasets]

    def send(self, actions: Sequence) -> List:
        return [d.send(action) for d, action in zip(self.datasets, actions)]
