""" Defines a Method, which is a "solution" for a given "problem" (a Setting).

The Method could be whatever you want, really. For the 'baselines' we have here,
we use pytorch-lightning, and a few little utility classes such as `Metrics` and
`Loss`, which are basically just like dicts/objects, with some cool other
methods.
"""
from abc import ABC
from collections import OrderedDict
from dataclasses import dataclass, is_dataclass
from pathlib import Path
from typing import (Any, ClassVar, Dict, Generic, List, Optional, Sequence,
                    Set, Tuple, Type, TypeVar, Union)

import torch
from common import Config, TrainerConfig, Loss, Metrics, Batch
from pytorch_lightning import (Callback, LightningDataModule, LightningModule,
                               Trainer)
from settings import Results, Setting, SettingType, Observations, Actions, Rewards, Environment
from simple_parsing import Serializable, mutable_field
from torch import Tensor
from torch.utils.data import DataLoader
from simple_parsing import mutable_field

from utils import Parseable, Serializable, get_logger
from utils.utils import get_path_to_source_file
from settings.method_abc import MethodABC
from .models.model import ForwardPass, Model

logger = get_logger(__file__)


@dataclass
class Method(MethodABC, Serializable, Parseable, ABC):
    """ A Method gets applied to a Setting to produce Results.

    A "Method", concretely, should consist of a LightningModule and a Trainer.
    - The model should accept a setting into its constructor.
    - Settings are LightningDataModules, and are used to create the
        train/val/test dataloaders.

    GOAL: The goal is to have a Method be applicable to a variety of Settings!
    We could even perhaps reuse the Method object on different Settings entirely!
    """
    # NOTE: these two fields are also used to create the command-line arguments.
    
    # HyperParameters of the method.
    hparams: Model.HParams = mutable_field(Model.HParams)
    # Options for the Trainer object.
    trainer_options: TrainerConfig = mutable_field(TrainerConfig)

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
        # IDEA: Could also pass some kind of proxy object from the Setting to
        # the method, and hide/delete some attributes whenever the method
        # shouldn't have access to them?
        # print(setting.dumps_json(indent="\t"))
        self.trainer: Trainer = self.create_trainer(setting)
        self.model: LightningModule = self.create_model(setting)

    def fit(self,
            train_dataloader: Environment[Observations, Actions, Rewards] = None,
            valid_dataloader: Environment[Observations, Actions, Rewards] = None,
            datamodule: LightningDataModule = None):
        """Called by the Setting to train the method.

        Might be called more than once before training is 'done'.
        Overwrite this to customize training.
        """
        assert self.model is not None, f"For now, Setting should have been nice enough to call method.configure(setting=self) before calling `fit`!"
        return self.trainer.fit(
            model=self.model,
            train_dataloader=train_dataloader,
            val_dataloaders=valid_dataloader,
            datamodule=datamodule,
        )

    def test(self,
             test_dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
             ckpt_path: Optional[str] = 'best',
             verbose: bool = True,
             datamodule: Optional[LightningDataModule] = None) -> Metrics:
        """ Test the method on the given test data and return the corresponding
        Metrics.

        TODO: It would be better if we had a more "closed" interface where we
        would just give the unlabeled samples and ask for predictions, and
        calculate the accuracy ourselves.
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
    
    def get_actions(self, observations: Observations) -> Actions:
        """ Get a batch of predictions (actions) for a batch of observations.
        
        This gets called by the Setting during the test loop.
        """
        self.model.eval()
        with torch.no_grad():
            forward_pass = self.model(observations)
        # Simplified this for now, but we could add more flexibility later.
        assert isinstance(forward_pass, ForwardPass)
        return forward_pass.actions

    def model_class(self, setting: SettingType) -> Type[LightningModule]:
        """ Returns the type of model to use for the given setting.
        
        You could extend this to customize which model is used depending on the
        setting.
        
        TODO: As @oleksost pointed out, this might allow the creation of weird
        'frankenstein' methods that are super-specific to each setting, without
        really having anything in common.
        """
        return Model

    def create_model(self, setting: SettingType) -> Model[SettingType]:
        """Creates the Model (a LightningModule) for the given Setting.

        The model should ideally accept a Setting in its constructor.
        
        IDEA: Would it be better if we could instead pass some kind of 'spec' so
        methods can't change things inside the Setting? 

        Args:
            setting (SettingType): An experimental setting.

        Returns:
            Model[SettingType]: The Model that is to be applied to that setting.
        """
        # Get the type of model to use for that setting.
        model_class: Type[Model] = self.model_class(setting)
        hparams_class = model_class.HParams
        logger.debug(f"model class for this setting: {model_class}")
        logger.debug(f"hparam class for this setting: {hparams_class}")
        logger.debug(f"hparam class on the method: {type(self.hparams)}")

        if isinstance(self.hparams, hparams_class):
            # Create the model, passing the setting and hparams.
            return model_class(setting=setting, hparams=self.hparams, config=self.config)
        else:
            # Need to 'upgrade' the hparams.
            # TODO: @lebrice This is ugly, and should be cleaned up somehow. Let
            # me know what you think:
            #
            # The problem is that in order to have the --help option display all
            # the options for the Method (including the model hparams), the
            # hparams should be one or more fields on the Method object.
            #
            # However, if in our method we use a different Model class depending
            # on the type of Setting, then we would need the hyperparameters to
            # be of the type required by the model!
            #
            # Therefore, here we upgrade `self.hparams` (if present) to the
            # right type (`model_class.HParams`)
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
        loggers = self.config.create_loggers()
        callbacks = self.create_callbacks(setting)
        trainer = self.trainer_options.make_trainer(
            loggers=loggers,
            callbacks=callbacks,
        )
        return trainer

    def create_callbacks(self, setting: SettingType) -> List[Callback]:
        # TODO: Add some callbacks here if you want.
        return []
    
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
    target_setting: ClassVar[Type["SettingABC"]] = None

    def __init_subclass__(cls, target_setting: Type[Setting] = None, **kwargs) -> None:
        """Called when creating a new subclass of Method.

        Args:
            target_setting (Type[Setting], optional): The target setting.
                Defaults to None, in which case the method will inherit the
                target setting of it's parent class.
        """
        if not is_dataclass(cls):
            logger.critical(UserWarning(
                f"The Method class {cls} should be decorated with @dataclass!\n"
                f"While this isn't strictly necessary for things to work, it is"
                f"highly recommended, as any dataclass-style class attributes "
                f"won't have the corresponding command-line arguments "
                f"generated, which can cause a lot of subtle bugs."
            ))
        
        if target_setting:
            cls.target_setting = target_setting
        elif hasattr(cls, "target_setting"):
            target_setting = cls.target_setting
        else:
            raise RuntimeError(
                f"You must either pass a `target_setting` argument to the "
                f"class statement or have a `target_setting` class variable "
                f"when creating a new subclass of {__class__}."
            )
        # Register this new method on the Setting.
        target_setting.register_method(cls)
        return super().__init_subclass__(**kwargs)

    def upgrade_hparams(self, new_type: Type[Model.HParams]) -> Model.HParams:
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
        new_hparams: Model.HParams = new_type.from_args(argv)
        logger.debug(f"Hparams for that type of model (from the method): {self.hparams}")
        logger.debug(f"Hparams for that type of model (from command-line): {new_hparams}")
        return new_hparams
    
    @classmethod
    def get_path_to_source_file(cls: Type) -> Path:
        return get_path_to_source_file(cls)

    def split_batch(self, batch: Any) -> Tuple[Batch, Batch]:
        return self.model.split_batch(batch)
    
    def on_task_switch(self, task_id: int) -> None:
        """
        TODO: Not sure if it makes sense to put this here. Might have to move
        it to Class/Task incremental or something like that.
        """
        if hasattr(self.model, "on_task_switch"):
            self.model.on_task_switch(task_id)
