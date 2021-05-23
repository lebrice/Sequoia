""" Defines a Method, which is a "solution" for a given "problem" (a Setting).

The Method could be whatever you want, really. For the 'baselines' we have here,
we use pytorch-lightning, and a few little utility classes such as `Metrics` and
`Loss`, which are basically just like dicts/objects, with some cool other
methods.
"""
import json
import operator
import warnings
from dataclasses import dataclass, is_dataclass, fields
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch
import wandb
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from simple_parsing import mutable_field
from wandb.wandb_run import Run

from sequoia.common import Config, TrainerConfig
from sequoia.common.spaces import Image
from sequoia.settings import RLSetting, SLSetting
from sequoia.settings.rl.continual import ContinualRLSetting
from sequoia.settings.assumptions.incremental import IncrementalAssumption
from sequoia.settings.base import Method
from sequoia.settings.base.environment import Environment
from sequoia.settings.base.objects import Actions, Observations, Rewards
from sequoia.settings.base.results import Results
from sequoia.settings.base.setting import Setting, SettingType
from sequoia.utils import Parseable, Serializable, compute_identity, get_logger
from sequoia.methods import register_method

from .models import BaselineModel, ForwardPass

logger = get_logger(__file__)


@register_method
@dataclass
class BaselineMethod(Method, Serializable, Parseable, target_setting=Setting):
    """ Versatile Baseline method which targets all settings.

    Uses pytorch-lightning's Trainer for training and LightningModule as model.

    Uses a [BaselineModel](methods/models/baseline_model/baseline_model.py), which
    can be used for:
    - Self-Supervised training with modular auxiliary tasks;
    - Semi-Supervised training on partially labeled batches;
    - Multi-Head prediction (e.g. in task-incremental scenario);
    """

    # NOTE: these two fields are also used to create the command-line arguments.
    # HyperParameters of the method.
    hparams: BaselineModel.HParams = mutable_field(BaselineModel.HParams)
    # Configuration options.
    config: Config = mutable_field(Config)
    # Options for the Trainer object.
    trainer_options: TrainerConfig = mutable_field(TrainerConfig)

    def __init__(
        self,
        hparams: BaselineModel.HParams = None,
        config: Config = None,
        trainer_options: TrainerConfig = None,
        **kwargs,
    ):
        """ Creates a new BaselineMethod, using the provided configuration options.

        Parameters
        ----------
        hparams : BaselineModel.HParams, optional
            Hyper-parameters of the BaselineModel used by this Method. Defaults to None.

        config : Config, optional
            Configuration dataclass with options like log_dir, device, etc. Defaults to
            None.

        trainer_options : TrainerConfig, optional
            Dataclass which holds all the options for creating the `pl.Trainer` which
            will be used for training. Defaults to None.

        **kwargs :
            If any of the above arguments are left as `None`, then they will be created
            using any appropriate value from `kwargs`, if present.

        ## Examples:
        ```
        method = BaselineMethod(hparams=BaselineModel.HParams(learning_rate=0.01))
        method = BaselineMethod(learning_rate=0.01) # Same as above

        method = BaselineMethod(config=Config(debug=True))
        method = BaselineMethod(debug=True) # Same as above

        method = BaselineMethod(hparams=BaselineModel.HParams(learning_rate=0.01),
                                config=Config(debug=True))
        method = BaselineMethod(learning_rate=0.01, debug=True) # Same as above
        ```
        """
        # TODO: When creating a Method from a script, like `BaselineMethod()`,
        # should we expect the hparams to be passed? Should we create them from
        # the **kwargs? Should we parse them from the command-line?

        # Get the type of hparams to use from the field's type annotation.
        hparam_field = [f for f in fields(self) if f.name == "hparams"][0]
        hparam_type = hparam_field.type

        # Option 2: Try to use the keyword arguments to create the hparams,
        # config and trainer options.
        if kwargs:
            logger.info(
                f"using keyword arguments {kwargs} to populate the corresponding "
                f"values in the hparams, config and trainer_options."
            )
            self.hparams = hparams or hparam_type.from_dict(
                kwargs, drop_extra_fields=True
            )
            self.config = config or Config.from_dict(kwargs, drop_extra_fields=True)
            self.trainer_options = trainer_options or TrainerConfig.from_dict(
                kwargs, drop_extra_fields=True
            )

        elif self._argv:
            # Since the method was parsed from the command-line, parse those as
            # well from the argv that were used to create the Method.
            # Option 3: Parse them from the command-line.
            # assert not kwargs, "Don't pass any extra kwargs to the constructor!"
            self.hparams = hparams or hparam_type.from_args(
                self._argv, strict=False
            )
            self.config = config or Config.from_args(self._argv, strict=False)
            self.trainer_options = trainer_options or TrainerConfig.from_args(
                self._argv, strict=False
            )

        else:
            # Option 1: Use the default values:
            self.hparams = hparams or hparam_type()
            self.config = config or Config()
            self.trainer_options = trainer_options or TrainerConfig()
        assert self.hparams
        assert self.config
        assert self.trainer_options

        if self.config.debug:
            # Disable wandb logging if debug is True.
            self.trainer_options.no_wandb = True

        # The model and Trainer objects will be created in `self.configure`.
        # NOTE: This right here doesn't create the fields, it just gives some
        # type information for static type checking.
        self.trainer: Trainer
        self.model: BaselineModel

        self.additional_train_wrappers: List[Callable] = []
        self.additional_valid_wrappers: List[Callable] = []
        
        self.setting: Setting

    def configure(self, setting: SettingType) -> None:
        """Configures the method for the given Setting.

        Concretely, this creates the model and Trainer objects which will be
        used to train and test a model for the given `setting`.

        Args:
            setting (SettingType): The setting the method will be evaluated on.

        TODO: For the Challenge, this should be some kind of read-only proxy to the
        actual Setting.
        """
        # Note: this here is temporary, just tinkering with wandb atm.
        method_name: str = self.get_name()

        # Set the default batch size to use, depending on the kind of Setting.
        if self.hparams.batch_size is None:
            if isinstance(setting, RLSetting):
                # Default batch size of 1 in RL
                self.hparams.batch_size = 1
            elif isinstance(setting, SLSetting):
                self.hparams.batch_size = 32
            else:
                warnings.warn(
                    UserWarning(
                        f"Dont know what batch size to use by default for setting "
                        f"{setting}, will try 16."
                    )
                )
                self.hparams.batch_size = 16
        # Set the batch size on the setting.
        setting.batch_size = self.hparams.batch_size

        # TODO: Should we set the 'config' on the setting from here?
        if setting.config and setting.config == self.config:
            pass
        elif self.config != Config():
            assert (
                setting.config is None or setting.config == Config()
            ), "method.config has been modified, and so has setting.config!"
            setting.config = self.config
        elif setting.config:
            assert (
                setting.config != Config()
            ), "Weird, both configs have default values.."
            self.config = setting.config

        setting_name: str = setting.get_name()
        dataset = setting.dataset

        if isinstance(setting, IncrementalAssumption):
            if self.hparams.multihead is None:
                # Use a multi-head model by default if the task labels are
                # available at training time and has more than one task.
                if setting.task_labels_at_test_time:
                    assert setting.task_labels_at_train_time
                self.hparams.multihead = setting.nb_tasks > 1

        if isinstance(setting, ContinualRLSetting):
            setting.add_done_to_observations = True
            if isinstance(setting.observation_space.x, Image):
                if self.hparams.encoder is None:
                    self.hparams.encoder = "simple_convnet"
                # TODO: Add 'proper' transforms for cartpole, specifically?
                from sequoia.common.transforms import Transforms

                transforms = [
                    Transforms.three_channels,
                    Transforms.to_tensor,
                    Transforms.resize_64x64,
                ]
                setting.transforms = transforms
                setting.train_transforms = transforms
                setting.val_transforms = transforms
                setting.test_transforms = transforms

            # Configure the baseline specifically for an RL setting.
            # TODO: Select which output head to use from the command-line?
            # Limit the number of epochs so we never iterate on a closed env.
            # TODO: Would multiple "epochs" be possible?
            if setting.max_steps is not None:
                self.trainer_options.max_epochs = 1
                self.trainer_options.limit_train_batches = setting.max_steps // (
                    setting.batch_size or 1
                )
                self.trainer_options.limit_val_batches = min(
                    setting.max_steps // (setting.batch_size or 1), 1000
                )
                # TODO: Test batch size is limited to 1 for now.
                # NOTE: This isn't used, since we don't call `trainer.test()`.
                self.trainer_options.limit_test_batches = setting.max_steps

        self.model = self.create_model(setting)
        assert self.hparams is self.model.hp

        # The PolicyHead actually does its own backward pass, so we disable
        # automatic optimization when using it.
        from .models.output_heads import PolicyHead

        if isinstance(self.model.output_head, PolicyHead):
            # Doing the backward pass manually, since there might not be a loss
            # at each step.
            self.trainer_options.automatic_optimization = False

        self.trainer = self.create_trainer(setting)
        self.setting = setting

    def fit(
        self,
        train_env: Environment[Observations, Actions, Rewards],
        valid_env: Environment[Observations, Actions, Rewards],
    ):
        """Called by the Setting to train the method.
        Could be called more than once before training is 'over', for instance
        when training on a series of tasks.
        Overwrite this to customize training.
        """
        assert self.model is not None, (
            "Setting should have been called method.configure(setting=self) "
            "before calling `fit`!"
        )
        # TODO: Figure out if there is a smarter way to reset the state of the Trainer,
        # rather than just creating a new one every time.
        self.trainer = self.create_trainer(self.setting)
        
        # NOTE: It doesn't seem sufficient to just do this, since for instance the
        # early-stopping callback would prevent training on future tasks, since they
        # have higher validation loss:
        # self.trainer.current_epoch = 0

        success = self.trainer.fit(
            model=self.model, train_dataloader=train_env, val_dataloaders=valid_env,
        )
        # BUG: After `fit`, it seems like the output head of the model is on the CPU?
        self.model.to(self.config.device)

        return success

    def get_actions(
        self, observations: Observations, action_space: gym.Space
    ) -> Actions:
        """ Get a batch of predictions (actions) for a batch of observations.

        This gets called by the Setting during the test loop.

        TODO: There is a mismatch here between the type of the output of this
        method (`Actions`) and the type of `action_space`: we should either have
        a `Discrete` action space, and this method should return ints, or this
        method should return `Actions`, and the `action_space` should be a
        `NamedTupleSpace` or something similar.
        Either way, `get_actions(obs, action_space) in action_space` should
        always be `True`.
        """
        self.model.eval()
        with torch.no_grad():
            forward_pass = self.model.forward(observations)
        actions: Actions = forward_pass.actions
        action_numpy = actions.actions_np
        assert action_numpy in action_space, (action_numpy, action_space)
        return actions

    def create_model(self, setting: SettingType) -> BaselineModel[SettingType]:
        """Creates the BaselineModel (a LightningModule) for the given Setting.

        You could extend this to customize which model is used depending on the
        setting.

        TODO: As @oleksost pointed out, this might allow the creation of weird
        'frankenstein' methods that are super-specific to each setting, without
        really having anything in common.

        Args:
            setting (SettingType): An experimental setting.

        Returns:
            BaselineModel[SettingType]: The BaselineModel that is to be applied
            to that setting.
        """
        # Create the model, passing the setting, hparams and config.
        return BaselineModel(setting=setting, hparams=self.hparams, config=self.config)

    def create_trainer(self, setting: SettingType) -> Trainer:
        """Creates a Trainer object from pytorch-lightning for the given setting.

        NOTE: At the moment, uses the KNN and VAE callbacks.
        To use different callbacks, overwrite this method.

        Args:

        Returns:
            Trainer: the Trainer object.
        """
        # We use this here to create loggers!
        callbacks = self.create_callbacks(setting)
        loggers = []
        if setting.wandb:
            wandb_logger = setting.wandb.make_logger()
            loggers.append(wandb_logger)
        trainer = self.trainer_options.make_trainer(
            config=self.config, callbacks=callbacks, loggers=loggers,
        )
        return trainer

    def get_experiment_name(self, setting: Setting, experiment_id: str = None) -> str:
        """Gets a unique name for the experiment where `self` is applied to `setting`.

        This experiment name will be passed to `orion` when performing a run of
        Hyper-Parameter Optimization.

        Parameters
        ----------
        - setting : Setting

            The `Setting` onto which this method will be applied. This method will be used when

        - experiment_id: str, optional

            A custom hash to append to the experiment name. When `None` (default), a
            unique hash will be created based on the values of the Setting's fields.

        Returns
        -------
        str
            The name for the experiment.
        """
        if not experiment_id:
            setting_dict = setting.to_dict()
            # BUG: Some settings have non-string keys/value or something?
            from sequoia.utils.utils import flatten_dict

            d = flatten_dict(setting_dict)
            experiment_id = compute_identity(size=5, **d)
        assert isinstance(
            setting.dataset, str
        ), "assuming that dataset is a str for now."
        return (
            f"{self.get_name()}-{setting.get_name()}_{setting.dataset}_{experiment_id}"
        )

    def get_search_space(self, setting: Setting) -> Mapping[str, Union[str, Dict]]:
        """Returns the search space to use for HPO in the given Setting.

        Parameters
        ----------
        setting : Setting
            The Setting on which the run of HPO will take place.

        Returns
        -------
        Mapping[str, Union[str, Dict]]
            An orion-formatted search space dictionary, mapping from hyper-parameter
            names (str) to their priors (str), or to nested dicts of the same form.
        """
        return {
            "hparams": self.hparams.get_orion_space(),
            "trainer_options": self.trainer_options.get_orion_space(),
        }

    def adapt_to_new_hparams(self, new_hparams: Dict[str, Any]) -> None:
        """Adapts the Method when it receives new Hyper-Parameters to try for a new run.

        It is required that this method be implemented if you want to perform HPO sweeps
        with Orion.
        
        Parameters
        ----------
        new_hparams : Dict[str, Any]
            The new hyper-parameters being recommended by the HPO algorithm. These will
            have the same structure as the search space.
        """
        # Here we overwrite the corresponding attributes with the new suggested values
        # leaving other fields unchanged.
        self.hparams = self.hparams.replace(**new_hparams["hparams"])
        # BUG with the `replace` function and Union[int, float] type, it doesn't
        # preserve the type of the field when serializing/deserializing!
        self.trainer_options.max_epochs = new_hparams["trainer_options"]["max_epochs"]

    def hparam_sweep(
        self,
        setting: Setting,
        search_space: Dict[str, Union[str, Dict]] = None,
        experiment_id: str = None,
        database_path: Union[str, Path] = None,
        max_runs: int = None,
        debug: bool = False,
    ) -> Tuple[BaselineModel.HParams, float]:
        # Setting max epochs to 1, just to keep runs somewhat short.
        # NOTE: Now we're actually going to have the max_epochs as a tunable
        # hyper-parameter, so we're not hard-setting this value anymore. 
        # self.trainer_options.max_epochs = 1
        
        # Call 'configure', so that we create `self.model` at least once, which will
        # update the hparams.output_head field to be of the right type. This is
        # necessary in order for the `get_orion_space` to retrieve all the hparams
        # of the output head.
        self.configure(setting)

        return super().hparam_sweep(
            setting=setting,
            search_space=search_space,
            experiment_id=experiment_id,
            database_path=database_path,
            max_runs=max_runs,
            debug = debug or self.config.debug,
        )

    def receive_results(self, setting: Setting, results: Results):
        """ Receives the results of an experiment, where `self` was applied to Setting
        `setting`, which produced results `results`.
        """
        # TODO: Reset the run name so a new one is used for each experiment.

    def create_callbacks(self, setting: SettingType) -> List[Callback]:
        """Create the PytorchLightning Callbacks for this Setting.

        These callbacks will get added to the Trainer in `create_trainer`.

        Parameters
        ----------
        setting : SettingType
            The `Setting` on which this Method is going to be applied.

        Returns
        -------
        List[Callback]
            A List of `Callaback` objects to use during training.
        """
        # TODO: Move this to something like a `configure_callbacks` method in the model,
        # once PL adds it.
        # from sequoia.common.callbacks.vae_callback import SaveVaeSamplesCallback
        return [
            EarlyStopping(monitor="val Loss")
            # self.hparams.knn_callback,
            # SaveVaeSamplesCallback(),
        ]

    def apply_all(
        self, argv: Union[str, List[str]] = None
    ) -> Dict[Type[Setting], Results]:
        """(WIP): Runs this Method on all its applicable settings.

        Returns
        -------

            Dict mapping from setting type to the Results produced by this method.
        """
        applicable_settings = self.get_applicable_settings()

        all_results: Dict[Type[Setting], Results] = {}
        for setting_type in applicable_settings:
            setting = setting_type.from_args(argv)
            results = setting.apply(self)
            all_results[setting_type] = results
        print(f"All results for method of type {type(self)}:")
        print(
            {
                method.get_name(): (results.get_metric() if results else "crashed")
                for method, results in all_results.items()
            }
        )
        return all_results

    def __init_subclass__(
        cls, target_setting: Type[SettingType] = Setting, **kwargs
    ) -> None:
        """Called when creating a new subclass of Method.

        Args:
            target_setting (Type[Setting], optional): The target setting.
                Defaults to None, in which case the method will inherit the
                target setting of it's parent class.
        """
        if not is_dataclass(cls):
            logger.critical(
                UserWarning(
                    f"The BaselineMethod subclass {cls} should be decorated with "
                    f"@dataclass!\n"
                    f"While this isn't strictly necessary for things to work, it is"
                    f"highly recommended, as any dataclass-style class attributes "
                    f"won't have the corresponding command-line arguments "
                    f"generated, which can cause a lot of subtle bugs."
                )
            )
        super().__init_subclass__(target_setting=target_setting, **kwargs)

    def on_task_switch(self, task_id: Optional[int]) -> None:
        """Called when switching between tasks.
        
        Args:
            task_id (int, optional): the id of the new task. When None, we are
            basically being informed that there is a task boundary, but without
            knowing what task we're switching to.
        """
        self.model.on_task_switch(task_id)

    def setup_wandb(self, run: Run) -> None:
        """ Called by the Setting when using Weights & Biases, after `wandb.init`.

        This method is here to provide Methods with the opportunity to log some of their
        configuration options or hyper-parameters to wandb.

        NOTE: The Setting has already set the `"setting"` entry in the `wandb.config` by
        this point.

        Parameters
        ----------
        run : wandb.Run
            Current wandb Run.
        """
        # TODO: (@lebrice) I think these will probably be set by the wandb logger,
        # run.config["config"] = self.config.to_dict()
        # Need to check wether this causes any issues.
        # run.config["hparams"] = self.hparams.to_dict()
        # run.config["trainer_config"] = self.trainer_options
