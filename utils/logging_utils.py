import inspect
import logging
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, TypeVar, Union

import torch.multiprocessing as mp
import tqdm

logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(name)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO,
)
logging.getLogger('simple_parsing').addHandler(logging.NullHandler())
root_logger = logging.getLogger("")
T = TypeVar("T")

def pbar(dataloader: Iterable[T], description: str="", *args, **kwargs) -> Iterable[T]:
    kwargs.setdefault("dynamic_ncols", True)
    pbar = tqdm.tqdm(dataloader, *args, **kwargs)
    if description:
        pbar.set_description(description)
    return pbar


def get_logger(name: str, level: int=None) -> logging.Logger:
    """ Gets a logger for the given file. Sets a nice default format. 
    TODO: figure out if we should add handlers, etc. 
    """
    name_is_path: bool = False
    try:
        p = Path(name)
        if p.exists():
            name = str(p.absolute().relative_to(Path.cwd()).as_posix())
            name_is_path = True
    except:
        pass
    from sys import argv
        
    logger = root_logger.getChild(name)
    if level is None and "-d" in argv or "--debug" in argv:
        level = logging.DEBUG
    if level is None:
        level = logging.INFO
    logger.setLevel(level)

    # if the name is already something like foo.py:256
    if not name_is_path and name[-1].isdigit():
        formatter = logging.Formatter('%(asctime)s, %(levelname)-8s log [%(name)s] %(message)s')
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        sh.setLevel(level)
        logger.addHandler(sh)
    # logger = logging.getLogger(name)
    tqdm_handler = TqdmLoggingHandler()
    tqdm_handler.setLevel(level)
    logger.addHandler(tqdm_handler)
    return logger
import sys

def log_calls(function: Callable, level=logging.INFO) -> Callable:
    """ Decorates a function and logs the calls to it and the passed args. """
    
    callerframerecord = inspect.stack()[1]    # 0 represents this line
                                            # 1 represents line at caller
    frame = callerframerecord[0]
    info = inspect.getframeinfo(frame)

    p = Path(info.filename)
    name = str(p.absolute().relative_to(Path.cwd()).as_posix())
    logger = get_logger(f"{name}:{info.lineno}")
    @wraps(function)
    def _wrapped(*args, **kwargs):
        process_name = mp.current_process().name
        logger.log(level, (
            f"Process {process_name} called {function.__name__} with "
            f"args={args} and kwargs={kwargs}."
        ))
        return function(*args, **kwargs)
    return _wrapped


def get_new_file(file: Path) -> Path:
    """Creates a new file, adding _{i} suffixes until the file doesn't exist.
    
    Args:
        file (Path): A path.
    
    Returns:
        Path: a path that is new. Might have a new _{i} suffix.
    """
    if not file.exists():
        return file
    else:
        i = 0
        file_i = file.with_name(file.stem + f"_{i}" + file.suffix)
        while file_i.exists():
            i += 1
            file_i = file.with_name(file.stem + f"_{i}" + file.suffix)
        file = file_i
    return file


def cleanup(message: Dict[str, Union[Dict, str, float, Any]], sep: str="/") -> Dict[str, Union[str, float, Any]]:
    """Cleanup a message dict before it is logged to wandb.

    Args:
        message (Dict[str, Union[Dict, str, float, Any]]): [description]
        sep (str, optional): [description]. Defaults to "/".

    Returns:
        Dict[str, Union[str, float, Any]]: [description]
    """
    # Flatten the log dictionary
    from utils.utils import flatten_dict
    
    message = flatten_dict(message, separator=sep)

    # TODO: Remove redondant/useless keys
    for k in list(message.keys()):
        if k.endswith((f"{sep}n_samples", f"{sep}name")):
            message.pop(k)
            continue

        v = message.pop(k)
        # Example input:
        # "Task_losses/Task1/losses/Test/losses/rotate/losses/270/metrics/270/accuracy"
        # Simplify the key, by getting rid of all the '/losses/' and '/metrics/' etc.
        things_to_remove: List[str] = [f"{sep}losses{sep}", f"{sep}metrics{sep}"]
        for thing in things_to_remove:
            while thing in k:
                k = k.replace(thing, sep)
        # --> "Task_losses/Task1/Test/rotate/270/270/accuracy"

        
        # Get rid of repetitive modifiers (ex: "/270/270" above)
        parts = k.split(sep)
        from utils.utils import unique_consecutive
        k = sep.join(unique_consecutive(parts))
        # Will become:
        # "Task_losses/Task1/Test/rotate/270/accuracy"
        message[k] = v
    return message


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)  
