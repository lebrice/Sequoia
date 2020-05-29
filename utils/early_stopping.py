from dataclasses import dataclass
from typing import Generator, Optional
import logging
logger = logging.getLogger(__file__)

from common.losses import LossInfo


@dataclass
class EarlyStoppingOptions:
    """ Options related to early stopping.
    
    If `patience` is a positive int, then training after `patience`
    epochs show a worse performance than the best one observed so far.
    """
    # Number of epochs with worse performance than the best seen so far to wait
    # for before stopping training.
    patience: int = 0
    # Minimal required improvement from the best performance seen so far in
    # order to consider a new performance as better.
    min_delta: float = 0.


def early_stopping(options: EarlyStoppingOptions, use_acc: bool=False) -> Generator[bool, LossInfo, None]:
    """Generator for early stopping. Yields wether or not convergence was reached.

    Args:
        options (EarlyStoppingOptions): Set of options for this hook. Defaults to 3.
        use_acc (bool, optional): If True, uses accuracy, else uses loss. Defaults to False.

    Yields:
        Generator[bool, LossInfo, None]: Wether or not the model converged yet.
    """
    best_valid_perf: Optional[float] = None
    counter = 0

    min_delta = options.min_delta
    patience = options.patience

    converged = False
    
    while True:
        val_loss_info = yield converged
        if not val_loss_info:
            break
        
        if not options.patience:
            continue
        
        from tasks.tasks import Tasks
        val_loss = val_loss_info.total_loss.item()
        supervised_metrics = val_loss_info.metrics.get(Tasks.SUPERVISED)
            
        if use_acc:
            assert supervised_metrics, "Can't use accuracy since there are no supervised metrics in given loss.."
            val_acc = supervised_metrics.accuracy

        if use_acc and (best_valid_perf is None or val_acc > (best_valid_perf + min_delta)):
            counter = 0
            best_valid_perf = val_acc
        elif not use_acc and (best_valid_perf is None or val_loss < (best_valid_perf - min_delta)):
            counter = 0
            best_valid_perf = val_loss
        else:
            counter += 1
            message = (
                f"Validation {'accuracy' if use_acc else 'loss'} hasn't "
                f"{'increased' if use_acc else 'decreased'} over the last "
                f"{counter} epochs."
            )
            logger.info(message)
            if counter == patience:
                converged = True

