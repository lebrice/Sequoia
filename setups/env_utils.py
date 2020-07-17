"""Utility functions used to manipulate generators.
"""
from .base import EnvironmentBase, ObservationType, ActionType, RewardType
from torch.utils.data import IterableDataset
from typing import List, Generator
class ZipEnvironments(EnvironmentBase[List[ObservationType], List[ActionType], List[RewardType]], IterableDataset):
    """TODO: Trying to create a 'batched' version of a Generator.
    """
    def __init__(self, *generators: EnvironmentBase[ObservationType, ActionType, RewardType]):
        self.generators = generators
    
    def __next__(self) -> List[ObservationType]:
        return list(next(gen) for gen in self.generators)
    
    def __iter__(self) -> Generator[List[ObservationType], List[ActionType], None]:
        iterators = (
            iter(g) for g in self.generators
        )
        while True:
            actions = yield next(self)

        values = yield from zip(*iterators)
    
    def send(self, actions: List[ActionType]) -> List[RewardType]:
        if actions is not None:
            assert len(actions) == len(self.generators)
            self.action = actions
        return [
            gen.send(action) for gen, action in zip(self.generators, actions)
        ]

