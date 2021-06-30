"""Runs a hyper-parameter tuning sweep, using Orion for HPO and wandb for visualization. 

# PREREQUISITES:


1.  (Optional): If you want to run the sweep on the monsterkong env:
    At the time of writing, the monsterkong repo is private. Once the challenge is out,
    it will most probably be made public. In the meantime, you'll need to ask
    @mattriemer for access to the MonsterKong_examples repo.

    ```
    pip install -e .[rl]
    ```

2.  Install the repo, along with the optional dependencies for Hyper-Parameter
    Optimization (HPO):

    ```console
    pip install -e .[hpo]
    ```

    NOTE: You can also fuse the two steps above with `pip install -e .[rl,hpo]`

3.  (Optional) Setup a database to hold the hyper-parameter configurations, following
    the [Orion database configuration documentation](https://orion.readthedocs.io/en/stable/install/database.html)

    The quickest way to get this setup is to run the `orion db setup` wizard, entering
    "pickleddb" as the database type:

    ```console
    $ orion db setup
    Enter the database type:  (default: mongodb) pickleddb
    Enter the database name:  (default: test) 
    Enter the database host:  (default: localhost)
    Default configuration file will be saved at: 
    /home/<your username>/.config/orion.core/orion_config.yaml
    ```

"""
import wandb
from sequoia.common import Config
from sequoia.methods.base_method import BaseMethod
from sequoia.settings import TraditionalSLSetting, Results, Setting
from sequoia.utils.logging_utils import get_logger

logger = get_logger(__file__)


if __name__ == "__main__":
    from simple_parsing import ArgumentParser

    ## Create the Setting:
    from sequoia.settings import RLSetting
    setting = RLSetting(dataset="monsterkong")
    
    # from sequoia.settings import TaskIncrementalSLSetting
    # setting = TaskIncrementalSLSetting(dataset="cifar10")
    
    ## Create the BaseMethod:
    # Option 1: Create the method manually:
    # method = BaseMethod()

    # Option 2: From the command-line:
    method, unused_args = BaseMethod.from_known_args()  # allow unused args.
    # parser = ArgumentParser(description=__doc__)
    # BaseMethod.add_argparse_args(parser, dest="method")
    # args, unused_args = parser.parse_known_args()
    # method: BaseMethod = BaseMethod.from_argparse_args(args, dest="method")

    # Search space for the Hyper-Parameter optimization algorithm.
    # NOTE: This is just a copy of the spaces that are auto-generated from the fields of
    # the `BaselineModel.HParams` class. You can change those as you wish though.
    search_space = {
        "learning_rate": "loguniform(1e-06, 1e-02, default_value=0.001)",
        "weight_decay": "loguniform(1e-12, 1e-03, default_value=1e-06)",
        "optimizer": "choices(['sgd', 'adam', 'rmsprop'], default_value='adam')",
        "encoder": "choices({'resnet18': 0.5, 'simple_convnet': 0.5}, default_value='resnet18')",
        "output_head": {
            "activation": "choices(['relu', 'tanh', 'elu', 'gelu', 'relu6'], default_value='tanh')",
            "dropout_prob": "uniform(0, 0.8, default_value=0.2)",
            "gamma": "uniform(0.9, 0.999, default_value=0.99)",
            "normalize_advantages": "choices([True, False])",
            "actor_loss_coef": "uniform(0.1, 1, default_value=0.5)",
            "critic_loss_coef": "uniform(0.1, 1, default_value=0.5)",
            "entropy_loss_coef": "uniform(0, 1, discrete=True, default_value=0)",
        },
    }
    best_hparams, best_results = method.hparam_sweep(
        setting, search_space=search_space, experiment_id="123"
    )

    print(f"Best hparams: {best_hparams}, best perf: {best_results}")
    # results = setting.apply(method, config=Config(debug=True))

