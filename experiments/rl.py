from experiments.experiment import Results, Method, Experiment
from setups.rl import RLSetting
from dataclasses import dataclass
from models.classifier import Classifier
from models.cl_classifier import ContinualClassifier
from common.losses import LossInfo
from simple_parsing import mutable_field, choice, ArgumentParser

@dataclass
class Results(Results):
    """Results of an RL Experiment. """
    hparams: Classifier.HParams
    test_loss: LossInfo


@dataclass
class RLMethod(Method[RLSetting]):
    """Base class for a Method that is to be applied to an RLSetting.
    """
    # HyperParameters of the LightningModule (RL Agent).
    hparams: ContinualClassifier.HParams = mutable_field(ContinualClassifier.HParams)

    def __post_init__(self):
        self.setting: RLSetting
        self.config: RL.Config

    def apply(self, setting: RLSetting) -> Results:
        """ Applies this method to the particular experimental setting. """
        if not isinstance(setting, RLSetting):
            raise RuntimeError(
                f"Can only apply this method on an RL setting or "
                f"on a setting which inherits from RLSetting! "
                f"(setting is of type {type(setting)})."
            )        
        self.setting = setting
        logger.debug(f"Setting: {self.setting}")
        logger.debug(f"Config: {self.config}")
        return super().apply(setting)


@dataclass
class RL(Experiment):
    """ RL Experiment. """
    # Experimental Setting.
    setting: RLSetting = mutable_field(RLSetting)
    # Experimental method.
    method: RLMethod = mutable_field(RLMethod)

    @dataclass
    class Config(Experiment.Config):
        """ Config of an IID Experiment.

        Could use this to add some more command-line arguments if needed.
        """

    # Configuration options for the Experiment.
    config: Config = mutable_field(Config)

    def launch(self):
        """ Simple IID Experiment. """
        logger.info(f"Starting the RL experiment with log dir: {self.config.log_dir}")
        self.method.configure(self.config)
        results = self.method.apply(setting=self.setting)
        save_results_path = self.config.log_dir / "results.json"
        results.save(save_results_path)
        logger.info(f"Saved results of experiment at path {save_results_path}")
        return results


if __name__ == "__main__":
    RL.main()
