from dataclasses import InitVar, dataclass

from experiment_base import ExperimentBase
from addons import (ExperimentWithKNN, ExperimentWithReplay, ExperimentWithVAE,
                    LabeledPlotRegionsAddon, TestTimeTrainingAddon)
# Load up the addons, each of which adds independent, useful functionality to the Experiment base-class.
# TODO: This might not be the cleanest/most elegant way to do it, but it's better than having files with 1000 lines in my opinion.
from simple_parsing import mutable_field


@dataclass  # type: ignore
class Experiment(ExperimentWithKNN,
                 ExperimentWithVAE,
                 TestTimeTrainingAddon,
                 LabeledPlotRegionsAddon,
                 ExperimentWithReplay):
    """ Class used to perform a given 'method' or evaluation/experiment.
    (ex: Mnist_iid, Mnist_continual, Cifar10, etc. etc.)
    
    To create a new experiment, subclass this class, and add/change what you
    need to customize in the Config class.
    """
    pass

    @dataclass
    class Config(ExperimentWithKNN.Config,
                 ExperimentWithReplay.Config):
        """ Describes the parameters of an experiment. """
        pass
    
    def __post_init__(self, *args, **kwargs):
        from dataclasses import fields
        print([f.name for f in fields(self)])
        super().__post_init__(*args, **kwargs)

    config: Config = mutable_field(Config)

    # # Experiment Config: non-tunable parameters specific to an experiment.
    # config: Config = mutable_field(Config)
    # # Model Hyper-parameters (tunable) settings.
    # hparams: Classifier.HParams = mutable_field(Classifier.HParams)
    # # State of the experiment (not parsed form command-line).
    # state: State = mutable_field(State, init=False)