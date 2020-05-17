from models.classifier import Classifier
from task_incremental import TaskIncremental
from dataclasses import dataclass
from models.cl_method_wrappers.ewc_wrapper import EWC_wrapper


@dataclass
class TaskIncrementalWithEWC(TaskIncremental):
    """ Evaluates the model in the same setting as the OML paper's Figure 3.
    """
    # The 'lambda' parameter from EWC.
    # TODO: (Fabrice) IDK what this parameter means, maybe read up the EWC paper again and add a better description?
    ewc_lamda = 10

    def init_model(self) -> Classifier:
        print("init model")
        model = self.get_model_for_dataset(self.dataset)
        model.to(self.config.device)
        model = EWC_wrapper(model, lamda=self.ewc_lamda, n_ways=10, device=self.config.device)
        #TODO: n_ways should be self.n_classes_per_task, but model outputs 10 way classifier instead of self.n_classes_per_task - way
        return model

if __name__ == "__main__":
    from simple_parsing import ArgumentParser
    parser = ArgumentParser()
    parser.add_arguments(TaskIncrementalWithEWC, dest="experiment")
    
    args = parser.parse_args()
    experiment: TaskIncremental = args.experiment
    
    from main import launch
    launch(experiment)
