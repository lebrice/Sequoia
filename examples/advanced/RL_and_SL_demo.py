""" Demo where we add the same regularization loss from the other examples, but
this time as an `AuxiliaryTask` on top of the BaselineMethod.

This makes it easy to create CL methods that apply to both RL and SL Settings!
"""

import copy
import random
import sys
from argparse import Namespace
from dataclasses import dataclass
from typing import ClassVar, List

import torch
from simple_parsing import ArgumentParser, field
from torch import Tensor

# This "hack" is required so we can run `python examples/custom_baseline_demo.py`
sys.path.extend([".", ".."])

from sequoia.common.config import Config, TrainerConfig
from sequoia.common.loss import Loss
from sequoia.methods import BaselineMethod
from sequoia.methods.aux_tasks import AuxiliaryTask, SimCLRTask
from sequoia.methods.models import BaselineModel, ForwardPass
from sequoia.settings import Setting, Environment, RLSetting
from sequoia.utils import camel_case, dict_intersection, get_logger

logger = get_logger(__file__)


class SimpleRegularizationAuxTask(AuxiliaryTask):
    """ Same regularization loss as in the previous examples, this time
    implemented as an `AuxiliaryTask`, which gets added to the BaselineModel,
    making it applicable to both RL and SL.
    
    This adds a CL regularizaiton loss to the BaselineModel.
    
    The most important methods of `AuxiliaryTask` is `get_loss`, which should
    return a `Loss` for the given forward pass and resulting rewards/labels.
    Take a look at the `AuxiliaryTask` class for more info.    
    """

    name: ClassVar[str] = "simple_regularization"

    @dataclass
    class Options(AuxiliaryTask.Options):
        """Hyper-parameters / configuration options of this auxiliary task."""

        # Coefficient used to scale this regularization loss before it gets
        # added to the 'base' loss of the model.
        coefficient: float = 0.01
        # Wether to use the absolute difference of the weights or the difference
        # in the `regularize` method below.
        use_abs_diff: bool = False
        # The norm term for the 'distance' between the current and old weights.
        distance_norm: int = 2

    def __init__(
        self,
        *args,
        name: str = None,
        options: "SimpleRegularizationAuxTask.Options" = None,
        **kwargs,
    ):
        super().__init__(*args, options=options, name=name, **kwargs)
        self.options: SimpleRegularizationAuxTask.Options
        self.previous_task: int = None
        # TODO: Figure out a clean way to persist this dict into the state_dict.
        self.previous_model_weights: Dict[str, Tensor] = {}
        self.n_switches: int = 0

    def get_loss(self, forward_pass: ForwardPass, y: Tensor = None) -> Loss:
        """Get a `Loss` for the given forward pass and resulting rewards/labels.
        
        Take a look at the `AuxiliaryTask` class for more info,

        NOTE: This is the same simplified version of EWC used throughout the
        other examples: the loss is the P-norm between the current weights and
        the weights as they were on the begining of the task.
        Also note, this particular example doesn't actually use the provided
        arguments.
        """
        if self.previous_task is None:
            # We're in the first task: do nothing.
            return Loss(name=self.name)

        old_weights: Dict[str, Tensor] = self.previous_model_weights
        new_weights: Dict[str, Tensor] = dict(self.model.named_parameters())

        loss = 0.0
        for weight_name, (new_w, old_w) in dict_intersection(new_weights, old_weights):
            loss += torch.dist(
                new_w, old_w.type_as(new_w), p=self.options.distance_norm
            )

        ewc_loss = Loss(name=self.name, loss=loss)
        return ewc_loss

    def on_task_switch(self, task_id: int) -> None:
        """ Executed when the task switches (to either a new or known task).
        """
        if not self.enabled:
            return
        if self.previous_task is None and self.n_switches == 0:
            logger.debug(f"Starting the first task, no update.")
            pass
        elif task_id is None or task_id != self.previous_task:
            logger.debug(
                f"Switching tasks: {self.previous_task} -> {task_id}: "
                f"Updating the 'anchor' weights."
            )
            self.previous_task = task_id
            self.previous_model_weights.clear()
            self.previous_model_weights.update(
                copy.deepcopy({k: v.detach() for k, v in self.model.named_parameters()})
            )
        self.n_switches += 1


class CustomizedBaselineModel(BaselineModel):
    @dataclass
    class HParams(BaselineModel.HParams):
        """ Hyper-parameters of our customized baseline model.
        """

        # Hyper-parameters of our simple new auxiliary task.
        simple_reg: SimpleRegularizationAuxTask.Options = field(
            default_factory=SimpleRegularizationAuxTask.Options
        )

    def __init__(
        self,
        setting: Setting,
        hparams: "CustomizedBaselineModel.HParams",
        config: Config,
    ):
        super().__init__(setting=setting, hparams=hparams, config=config)
        self.hp: CustomizedBaselineModel.HParams

        # Here we add our new auxiliary task:
        self.add_auxiliary_task(SimpleRegularizationAuxTask(options=self.hp.simple_reg))
        # You could also add other auxiliary tasks, for example SimCLR:
        # self.add_auxiliary_task(SimCLRTask(coefficient=1.))

        # Or, add replay buffers of some sort:
        self.replay_buffer: List = []

        # (...)


@dataclass
class CustomMethod(BaselineMethod, target_setting=Setting):
    """ Example methods which adds regularization to the baseline in RL and SL.
    
    This extends the `BaselineMethod` by adding the simple regularization
    auxiliary task defined above to the `BaselineModel`.
    
    NOTE: Since this class inherits from `BaselineMethod`, which targets the
    `Setting` setting, i.e. the "root" node, it is applicable to all settings,
    both in RL and SL. However, you could customize the `target_setting`
    argument above to limit this to any particular subtree (only SL, only RL,
    only when task labels are present, etc).
    """

    # Hyper-parameters of the customized Baseline Model used by this method.
    hparams: CustomizedBaselineModel.HParams = field(
        default_factory=CustomizedBaselineModel.HParams
    )

    def __init__(
        self,
        hparams: CustomizedBaselineModel.HParams = None,
        config: Config = None,
        trainer_options: TrainerConfig = None,
        **kwargs,
    ):
        super().__init__(
            hparams=hparams, config=config, trainer_options=trainer_options, **kwargs,
        )

    def create_model(self, setting: Setting) -> CustomizedBaselineModel:
        """ Creates the Model to be used for the given `Setting`. """
        return CustomizedBaselineModel(
            setting=setting, hparams=self.hparams, config=self.config
        )

    def configure(self, setting: Setting):
        """ Configure this Method before being trained / tested on this Setting.
        """
        super().configure(setting)

        # For example, change the value of the coefficient of our
        # regularization loss when in RL vs SL:
        if isinstance(setting, RLSetting):
            self.hparams.simple_reg.coefficient = 0.01
        else:
            self.hparams.simple_reg.coefficient = 1.0

    def fit(self, train_env: Environment, valid_env: Environment):
        """ Called by the Setting to let the Method train on a given task.
        
        You can do whatever you want with the train and valid
        environments. As it is currently, in most `Settings`, the valid
        environment will contain data from only the current task. (See issue at
        https://github.com/lebrice/Sequoia/issues/46 for more context).
        """
        return super().fit(train_env=train_env, valid_env=valid_env)

    @classmethod
    def add_argparse_args(cls, parser: ArgumentParser, dest: str = ""):
        """Adds command-line arguments for this Method to an argument parser.
        
        NOTE: This doesn't do anything differently than the base implementation,
        but it's included here just for illustration purposes.
        """
        # 'dest' is where the arguments will be stored on the namespace.
        dest = dest or camel_case(cls.__qualname__)
        # Add all command-line arguments. This adds arguments for all fields of
        # this dataclass.
        parser.add_arguments(cls, dest=dest)
        # You could add arguments here if you wanted to:
        # parser.add_argument("--foo", default=1.23, help="example argument")

    @classmethod
    def from_argparse_args(cls, args: Namespace, dest: str = ""):
        """ Create an instance of this class from the parsed arguments. """
        # Retrieve the parsed arguments:
        dest = dest or camel_case(cls.__qualname__)
        method: CustomMethod = getattr(args, dest)
        # You could retrieve other arguments like so:
        # foo: int = args.foo
        return method


def demo_manual():
    """ Apply the custom method to a Setting, creating both manually in code. """
    # Create any Setting from the tree:
    from sequoia.settings import TaskIncrementalRLSetting, TaskIncrementalSetting

    # setting = TaskIncrementalSetting(dataset="mnist", nb_tasks=5)  # SL
    setting = TaskIncrementalRLSetting(  # RL
        dataset="cartpole",
        train_task_schedule={
            0: {"gravity": 10, "length": 0.5},
            5000: {"gravity": 10, "length": 1.0},
        },
        max_steps=10_000,
    )

    ## Create the BaselineMethod:
    config = Config(debug=True)
    trainer_options = TrainerConfig(max_epochs=1)
    hparams = BaselineModel.HParams()
    base_method = BaselineMethod(
        hparams=hparams, config=config, trainer_options=trainer_options
    )

    ## Get the results of the baseline method:
    base_results = setting.apply(base_method, config=config)

    ## Create the CustomMethod:
    config = Config(debug=True)
    trainer_options = TrainerConfig(max_epochs=1)
    hparams = CustomizedBaselineModel.HParams()
    new_method = CustomMethod(
        hparams=hparams, config=config, trainer_options=trainer_options
    )

    ## Get the results for the 'improved' method:
    new_results = setting.apply(new_method, config=config)

    print(f"\n\nComparison: BaselineMethod vs CustomMethod")
    print("\n BaselineMethod results: ")
    print(base_results.summary())

    print("\n CustomMethod results: ")
    print(new_results.summary())


def demo_command_line():
    """ Run the same demo as above, but customizing the Setting and Method from
    the command-line.
    
    NOTE: Remember to uncomment the function call below to use this instead of
    demo_simple!
    """
    ## Create the `Setting` and the `Config` from the command-line, like in
    ## the other examples.
    parser = ArgumentParser(description=__doc__)

    ## Add command-line arguments for any Setting in the tree:
    from sequoia.settings import TaskIncrementalRLSetting, TaskIncrementalSetting

    # parser.add_arguments(TaskIncrementalSetting, dest="setting")
    parser.add_arguments(TaskIncrementalRLSetting, dest="setting")
    parser.add_arguments(Config, dest="config")

    # Add the command-line arguments for our CustomMethod (including the
    # arguments for our simple regularization aux task).
    CustomMethod.add_argparse_args(parser, dest="method")

    args = parser.parse_args()

    setting: ClassIncrementalSetting = args.setting
    config: Config = args.config

    # Create the BaselineMethod:
    base_method = BaselineMethod.from_argparse_args(args, dest="method")
    # Get the results of the BaselineMethod:
    base_results = setting.apply(base_method, config=config)

    ## Create the CustomMethod:
    new_method = CustomMethod.from_argparse_args(args, dest="method")
    # Get the results for the CustomMethod:
    new_results = setting.apply(new_method, config=config)

    print(f"\n\nComparison: BaselineMethod vs CustomMethod:")
    print(base_results.summary())
    print(new_results.summary())


if __name__ == "__main__":
    demo_manual()
    # demo_command_line()
