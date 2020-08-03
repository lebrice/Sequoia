import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, TypeVar, Union
import utils
import tqdm
import wandb

logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(name)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO,
)
# logging.getLogger('simple_parsing').addHandler(logging.NullHandler())
root_logger = logging.getLogger()
T = TypeVar("T")

def log_wandb(message, step, prefix=None, print_message=False):
    for k, v in message.items():
            if hasattr(v, 'to_log_dict'):
                message[k] = v.to_log_dict()
    message = cleanup(message, sep="/")
    if prefix:
        message = utils.add_prefix(message, prefix)
    wandb.log(message, step=step)
    if print_message:
        print(message)


def pbar(dataloader: Iterable[T], description: str="", *args, **kwargs) -> Iterable[T]:
    kwargs.setdefault("dynamic_ncols", True)
    pbar = tqdm.tqdm(dataloader, *args, **kwargs)
    if description:
        pbar.set_description(description)
    return pbar


def get_logger(name: str) -> logging.Logger:
        """ Gets a logger for the given file. Sets a nice default format. 
        TODO: figure out if we should add handlers, etc. 
        """
        try:
            p = Path(name)
            if p.exists():
                name = str(p.absolute().relative_to(Path.cwd()).as_posix())
        except:
            pass
        from sys import argv
            
        logger = root_logger.getChild(name)
        if "-d" in argv or "--debug" in argv:
            logger.setLevel(logging.DEBUG)
        # logger = logging.getLogger(name)
        # logger.addHandler(TqdmLoggingHandler())
        return logger


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
        if 'Task_losses' in k and 'accuracy' in k and not 'AUC' in k:
            k = k.replace('Task_losses', 'Task_accuracies')

        if 'Cumulative' in k and 'accuracy' in k and not 'AUC' in k:
            k = 'Task_accuracies/'+k
        
        if 'coefficient' in k:
            k = 'coefficients/'+k
        
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
