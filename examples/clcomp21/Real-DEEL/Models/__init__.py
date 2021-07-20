from Models.mlp import MLP
from Models.vgg7 import VGG7
from .resnet18 import Resnet18
from .baseline import Baseline
from .vgg7 import VGG7
from .actor_critic import ActorCritic
from .model import Model
import torch.optim as optim

model_types_map = {
    "baseline": Baseline,
    "resnet18": Resnet18,
    "vgg7": VGG7,
    "mlp": MLP,
}

optimizers_map = {
    "adam": (optim.Adam, {}),
    "sgd": (optim.SGD, {}),
    "rmsprop": (optim.RMSprop, {}),
}

# step param means scheduler will run every step otherwise every epoch
schedulers_map = {
    "exponential": (
        optim.lr_scheduler.ExponentialLR,
        {"gamma": 0.999999, "last_epoch": -1, "verbose": False, "step": False},
    ),
    "steplr": (
        optim.lr_scheduler.StepLR,
        {"gamma": 0.2, "step_size": 2, "step": False},
    ),
    "cyclic": (
        optim.lr_scheduler.CyclicLR,
        {
            "mode": "triangular",
            "base_lr": 0.0003,
            "max_lr": 0.002,
            "cycle_momentum": False,
            "step": True,
        },
    ),
    "lambdalr": (
        optim.lr_scheduler.LambdaLR,
        {"lr_lambda": lambda x: x, "verbose": False, "step": True},
    ),
}
