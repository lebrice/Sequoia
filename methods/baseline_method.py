""" Defines a Method, which is a "solution" for a given "problem" (a Setting).

The Method could be whatever you want, really. For the 'baselines' we have here,
we use pytorch-lightning, and a few little utility classes such as `Metrics` and
`Loss`, which are basically just like dicts/objects, with some cool other
methods.
"""
from collections import OrderedDict
from dataclasses import dataclass, is_dataclass
from pathlib import Path
from typing import (Any, ClassVar, Dict, Generic, List, Optional, Sequence,
                    Set, Tuple, Type, TypeVar, Union)

import gym
import torch
from pytorch_lightning import (Callback, LightningDataModule, LightningModule,
                               Trainer)
from simple_parsing import Serializable, mutable_field
from torch import Tensor
from torch.utils.data import DataLoader

from common import Batch, Config, Loss, Metrics, TrainerConfig
from common.callbacks import KnnCallback
from settings.active.rl import ContinualRLSetting
from settings.base.environment import Environment
from settings.base.objects import Actions, Observations, Rewards
from settings.base.results import Results
from settings.base.setting import Setting, SettingType
from settings.base import MethodABC
from utils import Parseable, Serializable, get_logger, singledispatchmethod
from utils.utils import get_path_to_source_file

from .models import ActorCritic, BaselineModel, ForwardPass

logger = get_logger(__file__)



@dataclass
class BaselineMethod(MethodABC, Serializable, Parseable, target_setting=Setting):
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
    # Options for the Trainer object.
    trainer_options: TrainerConfig = mutable_field(TrainerConfig)
    config: Config = mutable_field(Config)
    
    # Adds Options for a KNN classifier callback, which is used to evaluate
    # the quality of the representations on each test and val task after each
    # training epoch.
    # TODO: Debug/test this callback to make sure it still works fine.
    # TODO: Debug/test this callback to make sure it still works fine.
    knn_callback: KnnCallback = mutable_field(KnnCallback)
    
    def __post_init__(self):
        # The model and Trainer objects will be created in `self.configure`. 
        # NOTE: This right here doesn't create the fields, it just gives some
        # type information for static type checking.
        self.trainer: Trainer
        self.model: LightningModule

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
        self.trainer: Trainer = self.create_trainer(setting)
        self.model: LightningModule = self.create_model(setting)
        self.Observations: Type[Observations] = setting.Observations
        self.Actions: Type[Actions] = setting.Actions
        self.Rewards: Type[Rewards] = setting.Rewards

    def fit(self,
            train_env: Environment[Observations, Actions, Rewards] = None,
            valid_env: Environment[Observations, Actions, Rewards] = None,
            datamodule: LightningDataModule = None):
        """Called by the Setting to train the method.
        Could be called more than once before training is 'over', for instance
        when training on a series of tasks.
        Overwrite this to customize training.
        """
        assert self.model is not None, (
            f"For now, Setting should have been nice enough to call "
            "method.configure(setting=self) before calling `fit`!"
        )
        return self.trainer.fit(
            model=self.model,
            train_dataloader=train_env,
            val_dataloaders=valid_env,
            datamodule=datamodule,
        )

    def test(self,
             test_dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
             ckpt_path: Optional[str] = 'best',
             verbose: bool = True,
             datamodule: Optional[LightningDataModule] = None) -> Metrics:
        """ Test the method on the given test dataloader and return the corresponding
        Metrics.

        TODO: It would be better if we had a more "closed" interface where we
        would just give the unlabeled samples and ask for predictions, and
        calculate the accuracy ourselves.
        
        NOTE: This isn't used atm. The idea is that some setting might be fine
        with giving the full test dataloader to the method, rather than having
        to step through it like a gym environment.
        """
        test_results = self.trainer.test(
            model=self.model,
            test_dataloaders=test_dataloaders,
            ckpt_path=ckpt_path,
            verbose=verbose,
            datamodule=datamodule,
        )
        assert len(test_results) == 1
        assert "loss_object" in test_results[0]
        total_loss: Loss = test_results[0]["loss_object"]

    def get_actions(self, observations: Observations, action_space: gym.Space) -> Actions:
        """ Get a batch of predictions (actions) for a batch of observations.
        
        This gets called by the Setting during the test loop.
        """
        self.model.eval()
        with torch.no_grad():
            forward_pass = self.model(observations)
        # Simplified this for now, but we could add more flexibility later.
        assert isinstance(forward_pass, ForwardPass)
        return forward_pass.actions

    def model_class(self, setting: SettingType) -> Type[BaselineModel]:
        """ Returns the type of model to use for the given setting.
        
        You could extend this to customize which model is used depending on the
        setting.
        
        TODO: As @oleksost pointed out, this might allow the creation of weird
        'frankenstein' methods that are super-specific to each setting, without
        really having anything in common.
        """
        return BaselineModel

    def create_model(self, setting: SettingType) -> BaselineModel[SettingType]:
        """Creates the BaselineModel (a LightningModule) for the given Setting.

        The model should ideally accept a Setting in its constructor.
        
        IDEA: Would it be better if we could instead pass some kind of 'spec' so
        methods can't change things inside the Setting? 

        Args:
            setting (SettingType): An experimental setting.

        Returns:
            BaselineModel[SettingType]: The BaselineModel that is to be applied to that setting.
        """
        # Get the type of model to use for that setting.
        model_class: Type[BaselineModel] = self.model_class(setting)
        hparams_class = model_class.HParams
        logger.debug(f"model class for this setting: {model_class}")
        logger.debug(f"hparam class for this setting: {hparams_class}")
        logger.debug(f"hparam class on the method: {type(self.hparams)}")

        if isinstance(self.hparams, hparams_class):
            # Create the model, passing the setting and hparams.
            return model_class(setting=setting, hparams=self.hparams, config=self.config)
        else:
            # Need to 'upgrade' the hparams on the method to match those on the
            # model.
            # TODO: @lebrice This is ugly.
            logger.warning(UserWarning(
                f"The hparams attribute on the {self.get_name()} Method are of "
                f"type {type(self.hparams)}, while the HParams on the model "
                f"class are of type {hparams_class}!\n"
                f"This will try to 'upgrade' the hparams, using values "
                f"from the command-line."
            ))
            self.hparams = self.upgrade_hparams(hparams_class)
            logger.info(f"'Upgraded' hparams: {self.hparams}")

        assert isinstance(self.hparams, model_class.HParams)
        # TODO: Could it become a problem that pytorch-lightning uses 'datamodule'
        # and we use 'setting' as a key?
        return model_class(setting=setting, hparams=self.hparams, config=self.config)

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
            callbacks=callbacks,
        )
        return trainer
    
    def create_callbacks(self, setting: SettingType) -> List[Callback]:
        # TODO: Move this to something like a `configure_callbacks` method 
        # in the model, once PL adds it.
        from common.callbacks.vae_callback import SaveVaeSamplesCallback
        return [
            self.knn_callback,
            # SaveVaeSamplesCallback(),
        ]

    @classmethod
    def main(cls, argv: Optional[Union[str, List[str]]]=None) -> Results:
        from main import Experiment
        experiment: Experiment
        # Create the Method object from the command-line:
        method = cls.from_args(argv)
        # Then create the 'Experiment' from the command-line, which makes it
        # possible to choose between all the settings.
        experiment = Experiment.from_args(argv)
        # Set the method attribute to be the one parsed above.
        experiment.method = method
        results: Results = experiment.launch(argv)
        return results

    def apply_all(self, argv: Union[str, List[str]] = None) -> Dict[Type["Method"], Results]:
        applicable_settings = self.get_applicable_settings()

        all_results: Dict[Type[Setting], Results] = OrderedDict()
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
    
    def on_task_switch(self, task_id: int) -> None:
        """
        TODO: Not sure if it makes sense to put this here. Might have to move
        it to Class/Task incremental or something like that.
        """
        model = getattr(self, "model", None)
        if model:
            if hasattr(model, "on_task_switch"):
                model.on_task_switch(task_id)
