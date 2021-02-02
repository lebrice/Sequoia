"""Runs a hyper-parameter tuning sweep, using Orion for HPO and wandb for visualization. 
"""
import wandb
from sequoia.common import Config
from sequoia.methods.baseline_method import BaselineMethod
from sequoia.settings import IIDSetting, Results, Setting
from sequoia.utils.logging_utils import get_logger

logger = get_logger(__file__)


if __name__ == "__main__":
    from simple_parsing import ArgumentParser

    ## Create the Setting:
    from sequoia.settings import RLSetting

    setting = RLSetting(dataset="monsterkong")
    ## Create the BaselineMethod:
    # Option 1: Create the method manually:
    # method = BaselineMethod()

    # Option 2: From the command-line:
    method, unused_args = BaselineMethod.from_known_args()  # allow unused args.
    # parser = ArgumentParser(description=__doc__)
    # BaselineMethod.add_argparse_args(parser, dest="method")
    # args, unused_args = parser.parse_known_args()
    # method: BaselineMethod = BaselineMethod.from_argparse_args(args, dest="method")

    search_space = {
        "learning_rate": "loguniform(1e-06, 1e-02)",
        "weight_decay": "loguniform(1e-12, 1e-03)",
        "optimizer": "choices(['sgd', 'adam', 'rmsprop'], default_value='adam')",
        "encoder": "choices({'resnet18': 0.5, 'simple_convnet': 0.5}, default_value='resnet18')",
        "output_head": {
            "activation": "choices(['relu', 'tanh', 'elu', 'gelu', 'relu6'], default_value='tanh')",
            "dropout_prob": "uniform(0, 0.8)",
        },
    }
    best_hparams, best_results = method.hparam_sweep(
        setting, search_space=search_space, experiment_id="123"
    )

    print(f"Best hparams: {best_hparams}, best perf: {best_results}")
    # results = setting.apply(method, config=Config(debug=True))

