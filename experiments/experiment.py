"""Simple CL experiments.

TODO: Add the other supported scenarios from continuum here, since that would
probably be pretty easy:
- New Instances
- New Classes
- New Instances & Classes
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import (ClassVar, Dict, Generic, List, Optional, Type, TypeVar,
                    Union)

from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.loggers.base import LightningLoggerBase

from common.losses import LossInfo
from common.metrics import ClassificationMetrics, Metrics
from config.config import Config as BaseConfig
from config.trainer_config import TrainerConfig
from models.cl_classifier import ContinualClassifier
from models.classifier import Classifier
from setups.base import ExperimentalSetting
from setups.cl import ClassIncrementalSetting
from simple_parsing import ArgumentParser, mutable_field
from utils.json_utils import Serializable
from utils.logging_utils import get_logger

logger = get_logger(__file__)

Setting = TypeVar("Setting", bound=ExperimentalSetting)


@dataclass
class Results(Serializable):
    """ Represents the results of an experiment.
    
    Here you can define what the quantity to maximize/minize is.
    This could be helpful when doing Hyper-Parameter Optimization.
    """
    hparams: Classifier.HParams
    test_loss: LossInfo


@dataclass
class Method(Serializable, Generic[Setting]):
    """ A Method gets applied to a Setting to produce Results.
    
    A "Method", concretely, should consist of a Trainer and a LightningModule.
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
    # HyperParameters of the model.
    hparams: ContinualClassifier.HParams = mutable_field(ContinualClassifier.HParams)

    def __post_init__(self):
        # The config and the Trainer should be created in `self.configure()`.
        self.config: Experiment.Config
        self.trainer: Trainer

        # The Setting and the model should be constructed in `self.apply`. 
        self.setting: Setting
        self.model: LightningModule

    def configure(self, config: "Experiment.Config") -> None:
        """Configures the method using the values from `config`.

        Should be called before applying the `Method` on a `Setting`.

        Concretely, this creates the Trainer object (stored at `self.trainer`)
        which will eventually be used to train and test a model once the method
        is applied on a `Setting`.

        Args:
            config (Experiment.Config): The configuration options.
        """
        self.config = config
        self.trainer = self.create_trainer(self.config)

    def apply(self, setting: Setting) -> Results:
        """ Applies this method to the given experimental setting.
        
        Extend this class and overwrite this method to customize your experiments.        
        """
        # This should look something like:

        # 0. Make sure that the configure() method was already called and the 
        # Trainer object was already constructed.
        assert self.config and self.trainer
        # 1. Create a model for the given setting:
        self.model = self.create_model(setting)
        # 2. Train the method on the setting.
        self.train(model)
        # 3. Create the results.
        results: Results = self.test(model)
        return results

    def train(self, model: LightningModule) -> None:
        """Trains the model. Overwrite this to customize training.

        Args:
            model (LightningModule): A LightningModule to train.
        """
        self.trainer.fit(model)

    def test(self, model: LightningModule) -> Results:
        """Tests the model and returns the Results. 

        Overwrite this to customize testing.

        Args:
            model (LightningModule): The model that was trained.

        Returns:
            Results: the Results object.
        """
        # This uses the test_dataloader from the model.

        test_outputs = self.trainer.test(model, verbose=False)
        # NOTE: This here is a simple example that might not always make sense.
        # Extend/overwrite this method for you particular method. 
        test_loss: LossInfo = test_outputs[0]["loss_info"]
        return Results(
            hparams=self.hparams,
            test_loss=test_loss,
        )

    def create_trainer(self, config: "Experiment.Config") -> Trainer:
        """ Creates a Trainer object from pytorch lightning.

        To customize what type of Trainer is used, you could extend the
        Experiment.Config class and specify your own Trainer in that method.

        TODO: @lebrice This probably isn't the easiest/cleanest way of going
        about this. Will think about it a bit and gather some feedback.        
        
        Args:
            config (Experiment.Config, optional): The configuration options. If
                None, will use 

        Returns:
            Trainer: The Trainer object, with the added loggers and callbacks.
        """
        # we pass None so the default values are used.
        return config.create_trainer(loggers=None, callbacks=None)

    def create_model(self, setting: Setting) -> Classifier[Setting]:
        """Creates the Model (a LightningModule).

        The model should ideally accept a Setting in its constructor.

        Args:
            setting (Setting): The experimental setting.

        Returns:
            Classifier[Setting]: The Model that is to be applied to that setting.
        """
        model = ContinualClassifier(
            setting=setting,
            hparams=self.hparams,
            config=self.config,
        )
        return model


ExperimentType = TypeVar("ExperimentType", bound="Experiment")

@dataclass
class Experiment(Serializable, Generic[Setting]):
    """ Experiment base class """
    # Experimental Setting.
    setting: Setting
    # Experimental method.
    method: Method

    @dataclass
    class Config(BaseConfig):
        """ Configuration options for an experiment.

        Contains all the command-line arguments for things that aren't supposed
        to be hyperparameters, but still determine how and experiment takes
        place. For instance, things like wether or not CUDA is used, or where
        the log directory is, etc.

        Extend this class whenever you want to add some command-line arguments
        for your experiment.
        """

    config: Config = mutable_field(Config)

    def launch(self) -> Results:
        """ Applies the Method to the Setting to generate Results. """
        self.method.configure(self.config)
        results = self.method.apply(setting=self.setting)
        results.save(self.config.log_dir / "results.json")
        return results
    

    @classmethod
    def main(cls: Type[ExperimentType], argv: Optional[List[str]]=None) -> Results:
        parser = ArgumentParser(description=__doc__)
        parser.add_arguments(cls, dest="experiment")
        args = parser.parse_args(argv)
        experiment: ExperimentType = args.experiment
        return experiment.launch()


if __name__ == "__main__":
    Experiment.main()
