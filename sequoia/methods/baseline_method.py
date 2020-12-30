""" Defines a Method, which is a "solution" for a given "problem" (a Setting).

The Method could be whatever you want, really. For the 'baselines' we have here,
we use pytorch-lightning, and a few little utility classes such as `Metrics` and
`Loss`, which are basically just like dicts/objects, with some cool other
methods.
"""
import warnings
from dataclasses import dataclass, is_dataclass
from pathlib import Path
from typing import (Any, Callable, ClassVar, Dict, Generic, List, Optional,
                    Sequence, Set, Tuple, Type, TypeVar, Union)

import gym
import torch
import wandb
from pytorch_lightning import (Callback, LightningDataModule, LightningModule,
                               Trainer)
from pytorch_lightning.loggers import WandbLogger
from simple_parsing import Serializable, mutable_field
from torch import Tensor
from torch.utils.data import DataLoader

from sequoia.common import Batch, Config, Loss, Metrics, TrainerConfig
from sequoia.common.callbacks import KnnCallback
from sequoia.common.config import WandbLoggerConfig
from sequoia.common.gym_wrappers import (AddDoneToObservation,
                                         AddInfoToObservation)
from sequoia.settings.active.continual import ContinualRLSetting
from sequoia.settings.assumptions.incremental import IncrementalSetting
from sequoia.settings.base import Method
from sequoia.settings.base.environment import Environment
from sequoia.settings.base.objects import Actions, Observations, Rewards
from sequoia.settings.base.results import Results
from sequoia.settings.base.setting import Setting, SettingType
from sequoia.settings import PassiveSetting, ActiveSetting
from sequoia.utils import (Parseable, Serializable, get_logger,
                           singledispatchmethod)
from sequoia.utils.utils import get_path_to_source_file

from .models import BaselineModel, ForwardPass

logger = get_logger(__file__)

from sequoia.methods import register_method


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
    
    def __init__(self,
                 hparams: BaselineModel.HParams = None,
                 config: Config = None,
                 trainer_options: TrainerConfig = None, **kwargs):
        # TODO: When creating a Method from a script, like `BaselineMethod()`,
        # should we expect the hparams to be passed? Should we create them from
        # the **kwargs? Should we parse them from the command-line? 
        
        # Option 1: Use the default values:
        # self.hparams = hparams or BaselineModel.HParams()
        # self.config = config or Config()
        # self.trainer_options = trainer_options or TrainerConfig()
        
        # Option 2: Try to use the keyword arguments to create the hparams,
        # config and trainer options.
        # self.hparams = hparams or BaselineModel.HParams.from_dict(kwargs, drop_extra_fields=True)
        # self.config = config or Config.from_dict(kwargs, drop_extra_fields=True)
        # self.trainer_options = trainer_options or TrainerConfig.from_dict(kwargs, drop_extra_fields=True)
        
        # Option 3: Parse them from the command-line.
        assert not kwargs, "Don't pass any extra kwargs to the constructor!"
        self.hparams = hparams or BaselineModel.HParams.from_args(strict=False)
        self.config = config or Config.from_args(strict=False)
        self.trainer_options = trainer_options or TrainerConfig.from_args(strict=False)

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
      
    
    def configure(self, setting: SettingType) -> None:
        """Configures the method for the given Setting.

        Concretely, this creates the model and Trainer objects which will be
        used to train and test a model for the given `setting`.

        Args:
            setting (SettingType): The setting the method will be evaluated on.
        
        TODO: This might be a problem if we're gonna avoid 'cheating'.. we're
        essentially giving the 'Setting' object
        directly to the method.. so I guess the object could maybe 
        """
        # Note: this here is temporary, just tinkering with wandb atm.
        method_name: str = self.get_name()
        
        # Set the default batch size to use.
        if self.hparams.batch_size is None:
            if isinstance(setting, ActiveSetting):
                # Default batch size of 1 in RL
                self.hparams.batch_size = 1
            elif isinstance(setting, PassiveSetting):
                self.hparams.batch_size = 32
            else:
                warnings.warn(UserWarning(
                    f"Dont know what batch size to use by default for setting "
                    f"{setting}, will try 16."
                ))
                self.hparams.batch_size = 16

        # TODO: Should we set the 'config' on the setting from here?
        # setting.config = self.config
        if setting.config == self.config:
            pass
        elif self.config != Config():
            assert setting.config == Config(), "method.config has been modified, and so has setting.config!"
            setting.config == self.config
        else:
            assert setting.config != Config(), "Weird, both configs have default values.."
            self.config = setting.config
        
        setting_name: str = setting.get_name()
        dataset: str = setting.dataset
        wandb_options: WandbLoggerConfig = self.trainer_options.wandb
        if wandb_options.run_name is None:
            wandb_options.run_name = f"{method_name}-{setting_name}" + (f"-{dataset}" if dataset else "")

        # TODO: Debug multihead model in RL.
        if isinstance(setting, IncrementalSetting):
            if self.hparams.multihead is None:
                # Use a multi-head model by default if the task labels are
                # available at both train and test time.
                assert setting.task_labels_at_train_time
                self.hparams.multihead = setting.task_labels_at_test_time

        if isinstance(setting, ContinualRLSetting):
            if self.hparams.batch_size is None:
                # Using default batch size of 32, which is huge for RL!
                self.hparams.batch_size = 1

            # Configure the baseline specifically for an RL setting.
            # TODO: Select which output head to use from the command-line?
            # Limit the number of epochs so we never iterate on a closed env.
            # TODO: Would multiple "epochs" be possible? 
            if setting.max_steps is not None:
                self.trainer_options.max_epochs = 1
                self.trainer_options.limit_train_batches = setting.max_steps // (setting.batch_size or 1)
                self.trainer_options.limit_val_batches = min(setting.max_steps // (setting.batch_size or 1), 1000)
                # TODO: Test batch size is limited to 1 for now.
                # NOTE: This isn't used, since we don't call `trainer.test()`.
                self.trainer_options.limit_test_batches = setting.max_steps

        # Set the batch size on the setting.
        setting.batch_size = self.hparams.batch_size
        self.model = self.create_model(setting)

        # The PolicyHead actually does its own backward pass, so we disable
        # automatic optimization when using it.
        from .models.output_heads import PolicyHead
        if isinstance(self.model.output_head, PolicyHead):
            # Doing the backward pass manually, since there might not be a loss
            # at each step.
            self.trainer_options.automatic_optimization = False

        self.trainer = self.create_trainer(setting)

        # Save the types to use.
        self.Observations: Type[Observations] = setting.Observations
        self.Actions: Type[Actions] = setting.Actions
        self.Rewards: Type[Rewards] = setting.Rewards

    def fit(self,
            train_env: Environment[Observations, Actions, Rewards] = None,
            valid_env: Environment[Observations, Actions, Rewards] = None):
        """Called by the Setting to train the method.
        Could be called more than once before training is 'over', for instance
        when training on a series of tasks.
        Overwrite this to customize training.
        """
        assert self.model is not None, (
            "Setting should have been called method.configure(setting=self) "
            "before calling `fit`!"
        )
        return self.trainer.fit(
            model=self.model,
            train_dataloader=train_env,
            val_dataloaders=valid_env,
        )

    def get_actions(self, observations: Observations, action_space: gym.Space) -> Actions:
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

        # Check if the observation is batched or not. If it isn't, add a
        # batch dimension to the inputs, and later remove any batch
        # dimension from the produced actions before they get sent back to
        # the Setting.
        single_obs_space = self.model.observation_space

        model_inputs = observations
        if observations[0].shape == single_obs_space[0].shape:
            model_inputs = observations.with_batch_dimension()

        with torch.no_grad():
            forward_pass = self.model(model_inputs)
        # Simplified this for now, but we could add more flexibility later.
        assert isinstance(forward_pass, ForwardPass)

        # If the original observations didn't have a batch dimension,
        # Remove the batch dimension from the results.
        if observations[0].shape == single_obs_space[0].shape:
            forward_pass = forward_pass.remove_batch_dimension()

        model_outputs: Actions = forward_pass.actions
        actions = model_outputs.actions_np
        assert actions in action_space, (actions, action_space)
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
        trainer = self.trainer_options.make_trainer(
            config=self.config,
            callbacks=callbacks,
        )
        return trainer
    
    def receive_results(self, setting: Setting, results: Results):
        # Note: this here is temporary, just tinkering with wandb atm.
        
        method_name: str = self.get_name()
        setting_name: str = setting.get_name()
        dataset: str = getattr(setting, "dataset", "")
        if not (self.config.debug or self.trainer_options.fast_dev_run or self.trainer_options.no_wandb):
            wandb.summary["method"] = method_name
            wandb.summary["setting"] = setting_name
            if dataset:
                wandb.summary["dataset"] = dataset
            wandb.log(results.to_log_dict())
            wandb.log(results.make_plots())
        # Reset the run name so we create a new one next time we're applied on a
        # Setting.
        self.trainer_options.wandb.run_name = None
    
    def create_callbacks(self, setting: SettingType) -> List[Callback]:
        # TODO: Move this to something like a `configure_callbacks` method 
        # in the model, once PL adds it.
        from sequoia.common.callbacks.vae_callback import SaveVaeSamplesCallback
        return [
            # self.hparams.knn_callback,
            # SaveVaeSamplesCallback(),
        ]

    def apply_all(self, argv: Union[str, List[str]] = None) -> Dict[Type["Method"], Results]:
        applicable_settings = self.get_applicable_settings()

        all_results: Dict[Type[Setting], Results] = {}
        for setting_type in applicable_settings:
            setting = setting_type.from_args(argv)
            results = setting.apply(self)
            all_results[setting_type] = results
        print(f"All results for method of type {type(self)}:")
        print({
            method.get_name(): (results.get_metric() if results else "crashed")
            for method, results in all_results.items()
        })
        return all_results

    def __init_subclass__(cls, *args, **kwargs) -> None:
        """Called when creating a new subclass of Method.

        Args:
            target_setting (Type[Setting], optional): The target setting.
                Defaults to None, in which case the method will inherit the
                target setting of it's parent class.
        """
        if not is_dataclass(cls):
            logger.critical(UserWarning(
                f"The BaselineMethod subclass {cls} should be decorated with "
                f"@dataclass!\n"
                f"While this isn't strictly necessary for things to work, it is"
                f"highly recommended, as any dataclass-style class attributes "
                f"won't have the corresponding command-line arguments "
                f"generated, which can cause a lot of subtle bugs."
            ))
        super().__init_subclass__(*args, **kwargs)

    def upgrade_hparams(self, new_type: Type[BaselineModel.HParams]) -> BaselineModel.HParams:
        """Upgrades the current hparams to the new type, filling in the new
        values from the command-line.

        Args:
            new_type (Type[HParams]): Type of HParams to upgrade to.
            argv (Union[str, List[str]], optional): Command-line arguments to
            use to set the missing values. Defaults to None, in which case the
            values in `sys.argv` are used.

        Returns:
            HParams: [description]
        """
        argv = self._argv
        logger.debug(f"Current method was originally created from args {argv}")
        new_hparams: BaselineModel.HParams = new_type.from_args(argv)
        logger.debug(f"Hparams for that type of model (from the method): {self.hparams}")
        logger.debug(f"Hparams for that type of model (from command-line): {new_hparams}")
        return new_hparams

    def split_batch(self, batch: Any) -> Tuple[Batch, Batch]:
        return self.model.split_batch(batch)

    def on_task_switch(self, task_id: Optional[int]) -> None:
        model = getattr(self, "model", None)
        if model:
            if hasattr(model, "on_task_switch"):
                model.on_task_switch(task_id)
