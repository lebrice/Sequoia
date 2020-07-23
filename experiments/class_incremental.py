"""Simple CL experiments.

TODO: Add the other supported scenarios from continuum here, since that would
probably be pretty easy:
- New Instances
- New Classes
- New Instances & Classes
"""

from dataclasses import dataclass
from typing import ClassVar, Dict, List, Optional, Type, Union

from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.loggers.base import LightningLoggerBase

from common.losses import LossInfo
from common.metrics import ClassificationMetrics, Metrics
from config.trainer_config import TrainerConfig
from models.cl_classifier import ContinualClassifier
from setups.base import ExperimentalSetting
from setups.cl import ClassIncrementalSetting
from simple_parsing import ArgumentParser, mutable_field
from utils.json_utils import Serializable
from utils.logging_utils import get_logger

from .experiment import Experiment, Method
from .experiment import Results as BaseResults
from .experiment import Setting

logger = get_logger(__file__)

@dataclass
class Results(BaseResults):
    """ Results of a ClassIncremental CL experiment. """
    hparams: ContinualClassifier.HParams
    test_loss: LossInfo


@dataclass
class ClassIncrementalMethod(Method):
    """Method which is to be applied to a class incremental CL problem setting.
    """
    # HyperParameters of the LightningModule.
    hparams: ContinualClassifier.HParams = mutable_field(ContinualClassifier.HParams)

    def __post_init__(self):
        super().__post_init__()
        self.setting: ClassIncrementalSetting
        self.model: ContinualClassifier

    def apply(self, setting: ClassIncrementalSetting) -> Results:
        """ Applies this method to the particular experimental setting.
        
        Extend this class and overwrite this method to create a different method.        
        """
        if not isinstance(setting, ClassIncrementalSetting):
            raise RuntimeError(
                f"Can only apply this method on a ClassIncremental setting or "
                f"on a setting which inherits from ClassIncrementalSetting! "
                f"(setting is of type {type(setting)})."
            )
        # The example code from `experiment` is fine for now.

        # 0. Make sure that the configure() method was already called and the 
        # Trainer object was already constructed.
        assert self.config and self.trainer
        self.setting = setting
        # 1. Create a model for the given setting:
        model = self.create_model(setting)
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
        # Just a sanity check:
        assert model.setting is self.setting
        
        n_tasks = self.setting.nb_tasks
        logger.info(f"Number of tasks: {n_tasks}")

        for i in range(n_tasks):
            logger.info(f"Starting task #{i}")
            model.on_task_switch(i)
            assert model.setting.current_task_id == self.setting.current_task_id == i

            self.trainer.fit(model)

            test_outputs: List[Dict] = self.trainer.test(model)
            print(len(test_outputs))

    def create_model(self, setting: Setting) -> ContinualClassifier:
        """Creates the Model (a LightningModule).

        The model should ideally accept a Setting in its constructor.

        Args:
            setting (Setting): The experimental setting.

        Returns:
            ContinualClassifier: The Model that is to be applied to that setting.
        """
        model = ContinualClassifier(
            setting=setting,
            hparams=self.hparams,
            config=self.config,
        )
        return model


@dataclass
class SelfSupervisedMethod(ClassIncrementalMethod):
    """
    TODO: Where/How does the Self Supervised stuff fit into this hierarchy?
    TODO: Add all the stuff related to self-supervision here, and maybe also
    create a subclass of Classifier which adds all the auxiliary tasks stuff?
    """
    pass


@dataclass
class ClassIncremental(Experiment):
    """ Class Incremental setting. """
    # Experimental Setting.
    setting: ClassIncrementalSetting = mutable_field(ClassIncrementalSetting)
    # Experimental method.
    method: ClassIncrementalMethod = mutable_field(ClassIncrementalMethod)

    @dataclass
    class Config(Experiment.Config):
        """ Config of a ClassIncremental experiment.

        Could use this to add some more command-line arguments if needed.
        """
    # Configuration of the Experiment.
    config: Config = mutable_field(Config)

    def launch(self):
        """ Simple Class-Incremental CL Experiment. """
        self.method.configure(self.config)
        results = self.method.apply(setting=self.setting)
        results.save(self.config.log_dir / "results.json")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_arguments(ClassIncremental, "experiment")
    args = parser.parse_args()
    experiment: ClassIncremental = args.experiment
    experiment.launch()
