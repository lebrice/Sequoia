import tqdm
from typing import Iterable, TypeVar

T = TypeVar("T")
def pbar(dataloader: Iterable[T], description: str="", *args, **kwargs) -> Iterable[T]:
    kwargs.setdefault("dynamic_ncols", True)
    pbar = tqdm.tqdm(dataloader, *args, **kwargs)
    if description:
        pbar.set_description(description)
    return pbar
