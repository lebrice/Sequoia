from dataclasses import dataclass
from .experiment import Experiment
from simple_parsing import mutable_field
from setups.cl import ClassIncrementalSetting
from models.classifier import Classifier


@dataclass
class ClassIncremental(Experiment):
    """ Class Incremental setting. """
    @dataclass
    class Config(Experiment.Config):
        """ Config of a ClassIncremental experiment. """
        # Options related to the the experiment's experimental setting.
        setting: ClassIncrementalSetting.Config = mutable_field(ClassIncrementalSetting.Config)
    
    config: Config = mutable_field(Config)

    def run(self):
        """ Simple class-incremental CL """
        model = Classifier(hparams=self.hparams, config=self.config)
        trainer = self.config.make_trainer()
        for i in range(self.config.setting.nb_tasks):
            self.config.setting.current_task_id += 1
            trainer.fit(model)

        # Save to results dir.
        test_results = trainer.test()
        
        print(f"test results: {test_results}") 
