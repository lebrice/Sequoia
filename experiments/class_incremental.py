"""Simple CL experiments.

TODO: Add the other supported scenarios from continuum here, since that would
probably be pretty easy:
- New Instances
- New Classes
- New Instances & Classes
"""

from dataclasses import dataclass
from typing import ClassVar, Dict, List, Optional, Type, Union

from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers.base import LightningLoggerBase

from config.trainer_config import TrainerConfig
from setups.base import ExperimentalSetting
from setups.cl import ClassIncrementalSetting
from simple_parsing import ArgumentParser, mutable_field
from utils.json_utils import Serializable
from utils.logging_utils import get_logger
from common.losses import LossInfo
from common.metrics import ClassificationMetrics, Metrics
from .experiment import Experiment
from models.cl_classifier import ContinualClassifier

logger = get_logger(__file__)

@dataclass
class ClassIncrementalExperimentResults(Serializable):
    """ TODO: A class which represents the results expected of a ClassIncremental CL experiment. """
    average_final_accuracy: float


@dataclass
class ClassIncrementalMethod():
    """Method which is to be applied to a class incremental CL problem setting.
    
    A "Method", concretely, should be composed of a Trainer and a LightningModule.
    - The model should accept a setting into its constructor.
    - Settings are LightningDataModules, and are used to create the train/val/test dataloaders.
    
    The attributes here are all the configurable options / hyperparameters
    needed to make the method work.

    TODO: Should the Config attribute be on the experiment, or on the method?
    """
    # HyperParameters of the LightningModule (in this case a ContinualClassifier).
    hparams: ContinualClassifier.HParams = mutable_field(ContinualClassifier.HParams)
    # Options for the Trainer.
    trainer: TrainerConfig = mutable_field(TrainerConfig)

    def __post_init__(self):
        self.setting: ClassIncrementalSetting
        self.config: Experiment.Config

    def apply(self, setting: ClassIncrementalSetting, config: Experiment.Config) -> ClassIncrementalExperimentResults:
        """ Applies this method to the particular experimental setting.
        
        Extend this class and overwrite this method to create a different method.        
        """
        if not isinstance(setting, ClassIncrementalSetting):
            raise RuntimeError(
                f"Can only apply this method on a ClassIncremental setting or "
                f"on a setting which inherits from ClassIncrementalSetting! "
                f"(setting is of type {type(setting)})."
            )
        
        self.setting = setting
        self.config = config

        # TODO: Figure out a way to actually get this from the command-line
        logger.debug(f"Setting: {self.setting}")
        logger.debug(f"Config: {self.config}")

        trainer = self.create_trainer()
        model = self.create_model()

        n_tasks = self.setting.nb_tasks
        logger.info(f"Number of tasks: {n_tasks}")
        
        # Matrix of task losses:
        # loss_grid[i][j] gives the loss (and metrics) on task j after having
        # learned tasks 0 through i.

        loss_grid: List[List[Optional[LossInfo]]] = [
            [None for i in range(n_tasks)]
            for j in range(n_tasks)
        ]

        # Just a sanity check:
        assert model.setting is self.setting is setting

        for i in range(n_tasks):
            logger.info(f"Starting task #{i}")
            self.setting.current_task_id = i
            assert model.setting.current_task_id == i

            trainer.fit(model)

            test_outputs: List[Dict] = trainer.test(model)
            print(len(test_outputs))
            exit()
            test_output: Dict = test_outputs[0]

            # TODO: This `test_loss` contains the losses for each task because the
            # ClassIncrementalSetting returns all the test dataloaders in `test_dataloader`,
            # which makes it so PL evaluates on each dataloader.
            task_losses: List[LossInfo] = output["losses"]
            for j, task_loss in enumerate(test_losses):
                print(f"Task {j} test acc: {task_loss.accuracy}")
                loss_grid[i].append(task_loss)
            
            exit()

            # Try to test each task individually
            # TODO: Doesn't quite work, This seems to be a bit finicky to use.
            # for j in range(self.setting.nb_tasks):
            #     # TODO: this actually gets all 5 dataloaders, not just the one of the current task.
            #     self.setting.current_task_id = j
            #     # Need to clarify or fix this by creating a subclass of Classifier specifically for CL maybe.
            #     task_j_loader = model.test_dataloader()[j]
            #     task_j_output = trainer.test(model, task_j_loader)
            #     task_j_loss: LossInfo = task_j_output[0]["loss_info"]
            #     logger.info(f"Accuracy on task {j} after learning task {i}: {task_j_loss.accuracy:.2%}")
                
            #     loss_grid[i].append(task_j_loss.detach())
        
        results = self.create_results_from_outputs(loss_grid)
        print(f"test results: {results}")
        return results 

    def create_results_from_outputs(self, outputs: List[Dict]) -> ClassIncrementalExperimentResults:
        """ Create a ClassIncrementalExperimentResults object from the outputs of
        Trainer.fit().
        """
        raise NotImplementedError("TODO")
        print(f"Num outputs: {len(outputs)}")
        result_dict = outputs[0]
        total_test_loss: LossInfo = result_dict["loss_info"]
        
        # print(total_test_loss)
        print(total_test_loss.metrics)
        all_metrics = total_test_loss.all_metrics()
        for i in range(self.setting.nb_tasks):
            exit()


        for metric_name, metric in all_metrics.items():
            print(f"metric : {metric_name}, {metric}")
        
        average_final_loss = sum(loss_grid[-1], LossInfo())
        print(f"Average final lossinfo:")
        print(average_final_loss.dumps())

        return ClassIncrementalExperimentResults(

        )

    def create_trainer(self) -> Trainer:
        callbacks = self.create_callbacks()
        loggers = self.create_loggers()
        return self.trainer.make_trainer(loggers=loggers, callbacks=callbacks)

    def create_callbacks(self) -> List[Callback]:
        from experiments.callbacks.vae_callback import SaveVaeSamplesCallback
        from experiments.callbacks.knn_callback import KnnCallback
        return [
            SaveVaeSamplesCallback(),
            KnnCallback(),
        ]

    def create_loggers(self) -> Optional[Union[LightningLoggerBase, List[LightningLoggerBase]]]:
        if self.config.debug:
            logger = None
        else:
            logger = self.config.wandb.make_logger(self.config.log_dir_root)
        return logger

    def create_model(self) -> ContinualClassifier:
        model = ContinualClassifier(setting=self.setting, hparams=self.hparams, config=self.config)
        return model



@dataclass
class SelfSupervisedCLMethod(ClassIncrementalMethod):
    """TODO: Add all the stuff related to self-supervision here, and maybe also
    create a subclass of Classifier which adds all the auxiliary tasks!
    """
    pass


@dataclass
class ClassIncremental(Experiment):
    """ Class Incremental setting. """
    # Experimental Setting.
    setting: ClassIncrementalSetting = mutable_field(ClassIncrementalSetting)
    # Experimental method.
    method: SelfSupervisedCLMethod = mutable_field(SelfSupervisedCLMethod)

    @dataclass
    class Config(Experiment.Config):
        """ Config of a ClassIncremental experiment.

        Could use this to add some more command-line arguments if needed.
        """
    config: Config = mutable_field(Config)

    def run(self):
        """ Simple class-incremental CL """
        print("Starting to run the ClassIncremental experiment.")
        # TODO: Figure out a way to actually get this from the command-line
        print(f"Setting: {self.setting}")
        logger.debug(f"Setting: {self.setting}")
        
        results = self.method.apply(setting=self.setting, config=self.config)
        print(f"Experimental results: {results.dumps(indent=1)}")
        results.save(self.config.log_dir / "results.json")
            


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_arguments(ClassIncremental, "experiment")
    args = parser.parse_args()
    experiment: Experiment = args.experiment
    experiment.launch()
