import torch
import torch.multiprocessing as mp
from pathlib import Path
import logging
logger = mp.get_logger()
logger.setLevel(logging.DEBUG)
from typing import *
from torch import Tensor
from config import Config
import wandb
from models.classifier import Classifier
from utils.json_utils import JsonSerializable

from functools import singledispatch

class SaveTuple(NamedTuple):
    save_path: Path
    obj: object


class SaverWorker(mp.Process):
    def __init__(self, config: Config, q: mp.Queue):
        super().__init__()
        self.config = config
        self.q = q

    def run(self):
        print(f"Starting a background thread. (inside Worker)")
        logger.info(f"Config: {self.config}")
        # if config.use_wandb:
        #     wandb.init(project="falr", config=hp.as_dict, group=hp.md5, job_type='background')
        item = self.q.get()
        while item is not None:
            if isinstance(item, SaveTuple):
                self.save(item)
            item = self.q.get()

    def save(self, save_tuple: SaveTuple) -> None:
        obj: Any = save_tuple.obj
        save_path: Path = save_tuple.save_path
        logger.debug(f"Asked to save {type(obj)} object to path {save_path}")
        # Create the parent directory
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # delegate to the registered save method for the type of obj.
        save(obj, save_path)
        logger.debug(f"Done saving object to path {save_path}")


@singledispatch
def save(obj: object, save_path: Path) -> None:
    # Save to the .tmp file (such that if the saving crashes or is interrupted,
    # we don't leave the file in a corrupted state.)
    save_path_tmp = save_path.with_suffix(".tmp")
    with open(save_path_tmp, "wb") as f:
        torch.save(obj, f)
    save_path_tmp.replace(save_path)


@save.register
def save_json(obj: JsonSerializable, save_path: Path) -> None:
    obj.save_json(save_path)
