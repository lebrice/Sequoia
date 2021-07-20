from dataclasses import dataclass
from sequoia.common.hparams import HyperParameters, log_uniform
import os
import random
import torch
import numpy as np


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True


@dataclass
class BaseHParams(HyperParameters):
    """ Hyper-parameters of the demo model. """

    # method to use
    cl_method_name: str = "mixder"
    # Learning rate of the optimizer.
    learning_rate: float = 0.0003  # log_uniform(1e-6, 1e-2, default=0.0004)
    # L2 regularization coefficient.
    weight_decay: float = 1e-6  # log_uniform(1e-9, 1e-3, default=1e-6)

    # Maximum number of training epochs per task.
    max_epochs_per_task: int = 7
    # reload best model from task
    reload_best: bool = True
    early_stop: bool = False
    # Number of epochs with increasing validation loss after which we stop training.
    early_stop_patience: int = 3
    # decay of early stopping after each task
    early_stop_decay_per_task: float = 0.95
    # use the smooth loss to stop training when no improvement
    early_stop_train: bool = False
    early_stop_train_patience: int = 60
    
    #Bias control
    bic: bool=False
    bic_epochs: int = 2

    # decay of max_epochs after each task
    max_epochs_decay_per_task: float = 0.95
    # seed which can be tuned
    seed: int = 42
    # model name to select from models in the models folder
    model_type: str = "resnet18"
    # optimizer name to select an optimizer
    optimizer: str = "adam"
    # scheduler name
    scheduler_name: str = "exponential"
    # training batch size
    batch_size: int = 32
    #number of workers
    num_workers:int = 4
    # SL params
    sl_nb_tasks: int = 12
    # Discount factor RL
    gamma: float = 0.99
    # Coefficient for the entropy term in the loss formula.
    entropy_term_coefficient: float = 0.001
    # output directory (parent path)
    output_dir: str = "output/"
    # wandbwandb agent cl-comp/dark/w04pclk3
    wandb: bool = False
    wandb_logging: bool = False
    wandb_api: str = ""
    wandb_entity: str = ""
    wandb_project: str = ""
    wandb_run_name: str = ""
    # enable debug mode
    debug_mode: bool = False
    # enable submission mode for faster runs
    submission: bool = False
    # use scheduler
    use_scheduler: bool = False



    # buffer options
    priority_reservoir: bool = False
    balanced_reservoir: bool = True
    save_after_val_improved: bool = True
    save_from_epoch: int = 0
    wandb_logging_buffer: bool = False

