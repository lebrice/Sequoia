""" Defines a Method, which is a "solution" for a given "problem" (a Setting).

The Method could be whatever you want, really. For the 'baselines' we have here,
we use pytorch-lightning, and a few little utility classes such as `Metrics` and
`Loss`, which are basically just like dicts/objects, with some cool other
methods.
"""
import inspect
import os
from abc import ABC
from collections import OrderedDict
from dataclasses import dataclass, is_dataclass
from inspect import getsourcefile
from pathlib import Path
from typing import (ClassVar, Dict, Generic, List, Optional, Sequence, Set,
                    Tuple, Type, TypeVar, Union, Any)

import torch
from common.config import Config, TrainerConfig
from common.loss import Loss
from common.metrics import Metrics
from pytorch_lightning import (Callback, LightningDataModule, LightningModule,
                               Trainer)
from settings import Results, Setting, SettingType
from simple_parsing import Serializable, mutable_field
from torch import Tensor
from torch.utils.data import DataLoader
from utils import Parseable, camel_case, get_logger, remove_suffix, try_get
from utils.utils import get_path_to_source_file

from .models.model import Model, Observation, Reward
from .models.batch import Batch

logger = get_logger(__file__)


@dataclass
class Method(Serializable, Generic[SettingType], Parseable, ABC):
    """ A Method gets applied to a Setting to produce Results.

    A "Method", concretely, should consist of a LightningModule and a Trainer.
    - The model should accept a setting into its constructor.
    - Settings are LightningDataModules, and are used to create the
        train/val/test dataloaders.

    The attributes here are all the configurable options / hyperparameters
    of the method.
    TODO: Not sure if the arguments to the Trainer object should be considered
    hyperparameters, hence I'm not sure where to put the TrainerConfig object.
    For now I'll keep it in Config (as if they weren't hyperparameters).

    GOAL: The goal is to have a Method be applicable to a variety of Settings!
    We could even perhaps reuse the Method object on different Settings entirely!
    """
    # HyperParameters of the method.
    hparams: Model.HParams = mutable_field(Model.HParams)
    # Options for the Trainer object.
    trainer_options: TrainerConfig = mutable_field(TrainerConfig)

    # Class attribute that holds the setting this method was designed to target.
    target_setting: ClassVar[Optional[Type[Setting]]] = None
    # class attribute that lists all the settings this method is applicable for.
    _applicable_settings: ClassVar[Set[Type[Setting]]] = set()

    def __post_init__(self):
        # The model and Trainer objects will be created in `self.configure`. 
        # NOTE: This right here doesn't create the fields, it just gives some
        # type information for static type checking.
        self.trainer: Trainer
        self.model: LightningModule

    
    #NOTE: (@lebrice) Removing this, as it is basically a duplicate of the
    # `Setting.apply` method, and I'm also trying to avoid giving the Setting
    # object directly to the method, whenever possible.
    # def apply_to(self, setting: SettingType) -> Results:
    #     """ Applies this method to the particular experimental setting.

    #     Extend this class and overwrite this method to customize training.       
    #     """
    #     if not self.is_applicable(setting):
    #         raise RuntimeError(
    #             f"Can only apply methods of type {type(self)} on settings "
    #             f"that inherit from {type(self).target_setting}. "
    #             f"(Given setting is of type {type(setting)})."
    #         )

    #     # Seed everything first:
    #     self.config.seed_everything()
    #     # Create a model and a Trainer for the given setting:
    #     self.configure(setting)
    #     # TODO: get the config object to be somewhere else, I guess?
    #     # TODO: "Who's" responsability should it be to create a Trainer object?
    #     return setting.apply(
    #         method=self,
    #         config=self.config,
    #     )

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
        self.trainer = self.create_trainer(setting)
        self.model = self.create_model(setting)
        

    def fit(self,
            train_dataloader: Optional[DataLoader] = None,
            val_dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
            datamodule: Optional[LightningDataModule] = None):
        """Trains the method (and its model) on the setting.

        Overwrite this to customize training.
        """
        assert self.model is not None, f"For now, Setting should have been nice enough to call method.configure(setting=self) before calling `fit`!"
        return self.trainer.fit(
            model=self.model,
            train_dataloader=train_dataloader,
            val_dataloaders=val_dataloaders,
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
    
    def split_batch(self, batch: Any) -> Tuple[Batch, Batch]:
        return self.model.split_batch(batch)

    def get_prediction(self, inputs) -> Tensor:
        """ Get a batch of predictions for this batch of inputs.
        
        TODO: @lebrice How should we specify what the input batch should
        contain, without restricting how future subclasses implement this method
        too much? (I hate using **kwargs!)
        Maybe some kind of 'Observation' dataclass or namedtuple? Sounds a bit
        over-engineered, even for me..
        """
        self.model.eval()
        with torch.no_grad():
            inputs = self.model.Observation.from_inputs(inputs)
            forward_pass = self.model(inputs)
        
        if isinstance(forward_pass, Tensor):
            # Forward pass wasn't a dict, so we assume it's the right tensor.
            action = forward_pass
        else:
            # NOTE (@lebrice): I think it's reasonable to assume that the
            # 'prediction' can always be found at one of these keys. Worst comes
            # to worst, we could always add more keys to this list if needed.
            # TODO: We could also maybe get the right key to use from the
            # current setting! This sounds cleaner, design-wise.
            possible_keys: List[str] = ["y_pred", "actions", "action"]
            action = try_get(forward_pass, *possible_keys)
            if action is None:
                raise RuntimeError(
                    f"Unable to retrieve the 'prediction' from the results of "
                    f"the forward pass! (tried keys {possible_keys}) type(forward_pass) = {type(forward_pass)}"
                )
        return action

    def predict_image_labels(self,
                             x: Tensor,
                             task_labels: List[Optional[float]] = None) -> Tensor:
        """ When in a classification setting, give back the predicted labels for
        this batch of images.
        
        When `task_labels` is not None, then it is a list of optional floats.
        If `task_labels[i]` is an integer, then it indicates that `images[i]` is
        an image taken from the task with index `task_labels[i]`. If
        `task_labels[i]` is None, then the task label for that image isn't
        given.
        Naturally, when `task_labels` is None, no task labels are given.

        Your method should ideally leverage this information when available, as
        it makes the problem a lot easier!
        """
        actions = self.predict(x=x, task_labels=task_labels)
        
        return torch.argmax(y_pred, dim=-1)

    def on_task_switch(self, task_id: int) -> None:
        """
        TODO: Not sure if it makes sense to put this here. Might have to move
        it to Class/Task incremental or something like that.
        """
        if hasattr(self.model, "on_task_switch"):
            self.model.on_task_switch(task_id)

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
        logger.debug(f"Hyperparameters class on the method: {type(self.hparams)}")

        if isinstance(self.hparams, hparams_class):
            # All good. Just create the model, passing the setting and hparams.
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
    def is_applicable(cls, setting: Union[Setting, Type[Setting]]) -> bool:
        """Returns wether this Method is applicable to the given setting.

        A method is applicable on a given setting if and only if the setting is
        the method's target setting, or if it is a descendant of the method's
        target setting (below the target setting in the tree).
        
        Concretely, since the tree is implemented as an inheritance hierarchy,
        a method is applicable to any setting which is an instance (or subclass)
        of its target setting.

        Args:
            setting (SettingType): a Setting.

        Returns:
            bool: Wether or not this method is applicable on the given setting.
        """            
        assert cls.target_setting, f"Method {cls} has no target setting!"
        # NOTE: Setting is a subclass of LightningDataModule.
        if isinstance(setting, LightningDataModule):
            # if given a Setting or LightningDataModule object, get it's type.
            setting_type = type(setting)
        elif inspect.isclass(setting) and issubclass(setting, LightningDataModule):
            setting_type = setting
        else:
            raise RuntimeError(
                f"Invalid setting {setting}. Must be either an instance or a "
                f"subclass of Setting or LightningDataModule."
            )

        if not issubclass(setting_type, Setting):
            # If the given setting type is a LightningDataModule that doesn't
            # inherit from 'Setting' then we consider it the same way we would
            # an IID setting.
            from settings import IIDSetting
            setting_type = IIDSetting

        return issubclass(setting_type, cls.target_setting)
    
    @classmethod
    def get_all_applicable_settings(cls) -> List[Type[SettingType]]:
        from settings import Setting, all_settings
        return list(filter(cls.is_applicable, all_settings))

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

    def apply_all(self, argv: Union[str, List[str]]=None) -> Dict[Type["Method"], Results]:
        applicable_settings = self.get_all_applicable_settings()

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

    def __init_subclass__(cls, target_setting: Type[Setting]=None, **kwargs) -> None:
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
                f"don't have the corresponding command-line arguments "
                f"generated, which can cause a lot of subtle bugs."
            ))

        if target_setting:
            logger.debug(f"Method {cls} is designed for setting {target_setting}")
            cls.target_setting = target_setting
            cls.target_setting._applicable_methods.add(cls)
        else:
            logger.debug(f"Method {cls} didn't set a `target_setting` argument in the "
                         f"class constructor, using the target setting of the parent")
            target_setting = cls.target_setting

        assert target_setting, "You must specify a `setting` argument when creating a new Method!"
        cls._applicable_settings.add(target_setting)
        
        return super().__init_subclass__(**kwargs)
    
    @classmethod
    def get_name(cls) -> str:
        """ Gets the name of this method class. """
        name = getattr(cls, "name", None)
        if name is None:
            name = camel_case(cls.__qualname__)
            name = remove_suffix(name, "_method")
        return name

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
        logger.info(f"Current method was originally created from args {argv}")
        new_hparams: HParams = new_type.from_args(argv)
        logger.info(f"Hparams for that type of model (from the method): {self.hparams}")
        logger.info(f"Hparams for that type of model (from command-line): {new_hparams}")
        
        # if self.hparams:
        #     # IDEA: use some fancy dict comparisons to keep things that aren't the same
        #     # Not needed, because we saved the args that were used to create the instance.
        #     default_values = self.hparams.from_dict({})
        #     current_values = self.hparams.to_dict()
        #     different_values = utils.
        #     new_hparams = new_type.from_dict(hparams_dict, drop_extra_fields=True)
        return new_hparams
    
    @classmethod
    def get_path_to_source_file(cls: Type) -> Path:
        return get_path_to_source_file(cls)
