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
    parser = ArgumentParser(description=__doc__)
    BaselineMethod.add_argparse_args(parser, dest="method")
    parser.add_arguments(Config, dest="config")

    args = parser.parse_args()

    setting = IIDSetting(dataset="mnist")
    # setting: Setting = args.setting
    config: Config = args.config
    setting.config = config
    method = BaselineMethod.from_argparse_args(args, dest="method")
    best_hparams, best_results = method.hparam_sweep(setting)

    print(f"Best hparams: {best_hparams}, best perf: {best_results}")
    # results = setting.apply(method, config=Config(debug=True))


