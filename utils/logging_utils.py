import tqdm
from typing import Iterable, TypeVar, Dict, Union, Any

T = TypeVar("T")
def pbar(dataloader: Iterable[T], description: str="", *args, **kwargs) -> Iterable[T]:
    kwargs.setdefault("dynamic_ncols", True)
    pbar = tqdm.tqdm(dataloader, *args, **kwargs)
    if description:
        pbar.set_description(description)
    return pbar


def cleanup(message: Dict[str, Union[Dict, str, float, Any]],sep: str="/") -> Dict[str, Union[str, float, Any]]:
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
        k = k.replace(f"{sep}losses{sep}", sep).replace(f"{sep}metrics{sep}", sep)
        # --> "Task_losses/Task1/Test/rotate/270/270/accuracy"
        
        # Get rid of repetitive modifiers (ex: "/270/270" above)
        parts = k.split(sep)
        from utils.utils import unique_consecutive
        k = sep.join(unique_consecutive(parts))
        # Will become:
        # "Task_losses/Task1/Test/rotate/270/accuracy"
        message[k] = v
    return message
