"""Simple CL experiments.

TODO: Add the other supported scenarios from continuum here, since that would
probably be pretty easy:
- New Instances
- New Classes
- New Instances & Classes
"""

from dataclasses import dataclass
from typing import ClassVar, Type

from models.classifier import Classifier
from setups.base import ExperimentalSetting
from setups.cl import ClassIncrementalSetting
from simple_parsing import ArgumentParser, mutable_field
from utils.logging_utils import get_logger
from .experiment import Experiment
logger = get_logger(__file__)

@dataclass
class ClassIncremental(Experiment):
    """ Class Incremental setting. """
    # Experimental Setting.
    setting: ClassIncrementalSetting = mutable_field(ClassIncrementalSetting)
    
    @dataclass
    class Config(Experiment.Config):
        """ Config of a ClassIncremental experiment. """
        # setting_config: ExperimentalSetting.Config = mutable_field(ClassIncrementalSetting.Config)
        # setting_class: ClassVar[Type[ExperimentalSetting]] = ClassIncrementalSetting

    config: Config = mutable_field(Config)

    def run(self):
        """ Simple class-incremental CL """
        print("Starting to run the ClassIncremental experiment.")
        # TODO: Figure out a way to actually get this from the command-line
        print(f"Setting: {self.setting}")
        logger.debug(f"Setting: {self.setting}")
        model = Classifier(setting=self.setting, hparams=self.hparams, config=self.config)
        
        trainer = self.config.make_trainer()
        logger.info(f"Number of tasks: {self.setting.nb_tasks}")
        for i in range(self.setting.nb_tasks):
            logger.info(f"Starting task #{i}")
            logger.info(f"(setting current task id: {self.setting.current_task_id})")
            trainer.fit(model)

            self.setting.current_task_id += 1

        # Save to results dir.
        test_results = trainer.test()        
        print(f"test results: {test_results}") 


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_arguments(ClassIncremental, "experiment")
    args = parser.parse_args()
    experiment: Experiment = args.experiment
    experiment.launch()
