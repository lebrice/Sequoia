from models.classifier import Classifier
from task_incremental import TaskIncremental
from dataclasses import dataclass
from torch.utils.data import Subset
from datasets.subset import VisionDatasetSubset
from datasets.ss_dataset import get_sampler
from addons.ewc import EWC_wrapper
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, Dataset, TensorDataset
from typing import Dict, Iterable, List, Tuple, Union, Optional, Any


@dataclass
class TaskIncrementalWithEWC(TaskIncremental):
    """ Evaluates the model in the same setting as the OML paper's Figure 3.
    """
    # The 'lambda' parameter from EWC.
    # The factor in fron of the EWC regularizer  - higher lamda -> more penalty for changing the parameters
    ewc_lamda = 10

    def init_model(self) -> Classifier:
        print("init model")
        model = self.get_model_for_dataset(self.dataset)
        model.to(self.config.device)
        model = EWC_wrapper(model, lamda=self.ewc_lamda, n_ways=10, device=self.config.device)
        #TODO: n_ways should be self.n_classes_per_task, but model outputs 10 way classifier instead of self.n_classes_per_task - way
        return model

    def load_datasets(self, tasks: List[List[int]]) -> List[List[int]]:
        """Create the train, valid and cumulative datasets for each task.

        Returns:
            List[List[int]]: The groups of classes for each task.
        """
        # download the dataset.
        self.train_dataset, self.valid_dataset = self.dataset.load(data_dir=self.config.data_dir)
        self.train_loader = self.get_dataloader(self.train_dataset)
        self.valid_loader = self.get_dataloader(self.valid_dataset)
        assert self.dataset.train is not None
        assert self.dataset.valid is not None

        # safeguard the entire training dataset.
        train_full_dataset = self.train_dataset
        valid_full_dataset = self.valid_dataset

        self.train_datasets.clear()
        self.valid_datasets.clear()

        for i, task in enumerate(tasks):
            train = VisionDatasetSubset(train_full_dataset, task)
            valid = VisionDatasetSubset(valid_full_dataset, task)

            sampler_train, sampler_train_unlabelled = get_sampler(train.targets,n=100)
            sampler_valid, sampler_valid_unlabelled = get_sampler(valid.targets, n=100)

            self.train_datasets.append((train,sampler_train,sampler_train_unlabelled))
            self.valid_datasets.append((valid,sampler_valid,sampler_valid_unlabelled))

        # Use itertools.accumulate to do the summation of validation datasets.
        self.valid_cumul_datasets = list(accumulate(self.valid_datasets))

        for i, (train, valid, cumul) in enumerate(zip(self.train_datasets,
                                                      self.valid_datasets,
                                                      self.valid_cumul_datasets)):
            self.save_images(i, train, prefix="train_")
            self.save_images(i, valid, prefix="valid_")
            self.save_images(i, cumul, prefix="valid_cumul_")

        return tasks

    def get_dataloader(self, dataset: Dataset, sampler_labeller: SubsetRandomSampler, sampler_unlabelled: SubsetRandomSampler ) -> Tuple[DataLoader,DataLoader]:
        loader_train_labelled =  DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            sampler=sampler_labeller,
            num_workers=self.config.num_workers,
            pin_memory=self.config.use_cuda,
        )
        loader_train_unlabelled = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            sampler=sampler_unlabelled,
            num_workers=self.config.num_workers,
            pin_memory=self.config.use_cuda,
        )
        #TODO: addapt run to deal with this tuple of loaders
        return (loader_train_labelled, loader_train_unlabelled)


    def train_until_convergence(self, train_dataset: Dataset,
                                      valid_dataset: Dataset,*args, **kwargs):
        super().train_until_convergence(train_dataset, valid_dataset, *args, **kwargs)
        if self.config.debug:
            self.model.current_task_loader = self.get_dataloader(Subset(train_dataset, range(200)))
        else:
            self.model.current_task_loader = self.get_dataloader(train_dataset)

if __name__ == "__main__":
    from simple_parsing import ArgumentParser
    parser = ArgumentParser()
    parser.add_arguments(TaskIncrementalWithEWC, dest="experiment")
    
    args = parser.parse_args()
    experiment: TaskIncremental = args.experiment
    
    from main import launch
    launch(experiment)
