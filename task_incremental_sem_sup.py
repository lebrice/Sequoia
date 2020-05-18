
import tqdm
from sys import getsizeof
from torch import Tensor, nn
from itertools import repeat, cycle
from models.classifier import Classifier
from task_incremental import TaskIncremental
from dataclasses import dataclass
from torch.utils.data import Subset
from datasets.subset import VisionDatasetSubset
from common.losses import LossInfo
from datasets.ss_dataset import get_semi_sampler
from addons.ewc import EWC_wrapper
from collections import OrderedDict, defaultdict
from itertools import accumulate
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, Dataset, TensorDataset
from typing import Dict, Iterable, List, Tuple, Union, Optional, Any
from typing import (Any, ClassVar, Dict, Generator, Iterable, List, Optional, Tuple, Type, Union)
from common.losses import LossInfo, TrainValidLosses
from simple_parsing import mutable_field, list_field

@dataclass
class TaskIncrementalWithEWC(TaskIncremental):
    """ Evaluates the model in the same setting as the OML paper's Figure 3.
    """
    unsupervised_epochs_per_task: int = 5
    # The 'lambda' parameter from EWC.
    # The factor in fron of the EWC regularizer  - higher lamda -> more penalty for changing the parameters
    use_ewc = True
    ewc_lamda = 10
    # Container for train/valid losses that are logged periodically.
    all_losses: TrainValidLosses = mutable_field(TrainValidLosses)

    #labeled samples ratio
    ratio_labelled = 0.2

    def init_model(self) -> Classifier:
        print("init model")
        model = self.get_model_for_dataset(self.dataset)
        model.to(self.config.device)
        if self.use_ewc:
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
        #self.train_loader = self.get_dataloader(self.train_dataset)
        #self.valid_loader = self.get_dataloader(self.valid_dataset)
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

            sampler_train, sampler_train_unlabelled = get_semi_sampler(train.targets,p=self.ratio_labelled)
            sampler_valid, sampler_valid_unlabelled = get_semi_sampler(valid.targets, p=self.ratio_labelled)

            self.train_datasets.append((train,sampler_train,sampler_train_unlabelled))
            self.valid_datasets.append((valid,sampler_valid,sampler_valid_unlabelled))

        # Use itertools.accumulate to do the summation of validation datasets.
        self.valid_cumul_datasets = list(accumulate(self.valid_datasets))

        for i, (train, valid, cumul) in enumerate(zip(self.train_datasets,
                                                      self.valid_datasets,
                                                      self.valid_cumul_datasets)):
            self.save_images(i, train[0], prefix="train_")
            self.save_images(i, valid[0], prefix="valid_")
            self.save_images(i, cumul[0], prefix="valid_cumul_")

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


    def run(self):
        """Evaluates a model/method in the classical "task-incremental" setting.

        NOTE: We evaluate the performance on all tasks:
        - When the task has NOT been trained on before, we evaluate the ability
        of the representations to "generalize" to unseen tasks by training a KNN
        classifier on the representations of target task's training set, and
        evaluating it on the representations of the target task's validation set.
        - When the task has been previously trained on, we evaluate the
        classification loss/metrics (and auxiliary tasks, if any) as well as the
        representations with the KNN classifier.

        Roughly equivalent to the following pseudocode:
        ```
        # Training and Validdation datasets
        train_datasets: Dataset[n_tasks]
        valid_datasets: Dataset[n_tasks]

        # Arrays containing the loss/performance metrics. (Value at idx (i, j)
        # is the preformance on task j after having trained on tasks [0:i].)

        knn_losses: LossInfo[n_tasks][n_tasks]
        tasks_losses: LossInfo[n_tasks][j]  #(this is a lower triangular matrix)

        # Array of objects containing the loss/performance on tasks seen so far.
        cumul_losses: LossInfo[n_tasks]

        for i in range(n_tasks):
            train_until_convergence(train_datasets[i], valid_datasets[i])

            # Cumulative (supervised) validation performance.
            cumul_loss = LossInfo()

            for j in range(n_tasks):
                # Evaluate the representations with a KNN classifier.
                knn_loss_j = evaluate_knn(train_dataset[j], valid_datasets[j])
                knn_losses[i][j] = knn_loss_j

                if j <= i:
                    # We have previously trained on this class.
                    loss_j = evaluate(valid_datasets[j])
                    task_losses[i][j] = loss_j
                    cumul_loss += loss_j

            cumul_losses[i] = cumul_loss
        ```
        """
        self.model = self.init_model()

        if (self.started or self.restore_from_path) and not self.config.debug:
            self.logger.info(f"Experiment was already started in the past.")
            self.restore_from_path = self.checkpoints_dir / "state.json"
            self.logger.info(f"Will load state from {self.restore_from_path}")
            self.load_state(self.restore_from_path)

        if self.done:
            self.logger.info(f"Experiment is already done.")
            # exit()

        if self.state.global_step == 0:
            self.logger.info("Starting from scratch!")
            self.state.tasks = self.create_tasks_for_dataset(self.dataset)

        self.tasks = self.state.tasks
        if not self.config.debug:
            self.save()

        # Load the datasets
        self.load_datasets(self.tasks)
        self.n_tasks = len(self.tasks)

        self.logger.info(f"Class Ordering: {self.state.tasks}")

        if self.state.global_step == 0:
            self.state.knn_losses = [[None] * self.n_tasks] * self.n_tasks  # [N,N]
            self.state.task_losses = [[None] * (i + 1) for i in range(self.n_tasks)]  # [N,J]
            self.state.cumul_losses = [None] * self.n_tasks  # [N]

        for i in range(self.state.i, self.n_tasks):
            self.state.i = i
            self.logger.info(f"Starting task {i} with classes {self.tasks[i]}")

            # If we are using a multihead model, we give it the task label (so
            # that it can spawn / reuse the output head for the given task).
            if self.multihead:
                self.model.current_task_id = i



            # Training and validation datasets for task i.
            train_i, sampler_train_i, sampler_unlabelled_i = self.train_datasets[i]
            valid_i, sampler_valid_i, sampler_valid_unlabelled_i = self.valid_datasets[i]

            # EWC_specific: pass EWC_rapper the loader to compute fisher
            #call befor task change
            #====================
            if self.use_ewc:
                if self.config.debug:
                    sampler_train_, sampler_train_unlabelled_ = get_semi_sampler(Subset(train_i, range(200)).dataset.targets[:200], p=self.ratio_labelled)
                    self.model.current_task_loader = self.get_dataloader(Subset(train_i, range(200)), sampler_train_, sampler_train_unlabelled_)[0]
                else:
                    self.model.current_task_loader = self.get_dataloader(train_i, sampler_train_i, sampler_unlabelled_i)[0]
            #====================

            with self.plot_region_name(f"Learn Task {i}"):
                # We only train (unsupervised) if there is at least one enabled
                # auxiliary task and if the maximum number of unsupervised
                # epochs per task is greater than zero.
                self_supervision_on = any(task.enabled for task in self.model.tasks.values())

                if self_supervision_on and self.unsupervised_epochs_per_task:
                    # Temporarily remove the labels.
                    with train_i.without_labels(), valid_i.without_labels():
                        # Un/self-supervised training on task i.
                        self.state.all_losses += self.train_until_convergence(
                            (train_i,sampler_train_i,sampler_unlabelled_i),
                            (valid_i,sampler_valid_i,sampler_valid_unlabelled_i),
                            max_epochs=self.unsupervised_epochs_per_task,
                            description=f"Task {i} (Unsupervised)",
                        )

                # Train (supervised) on task i.
                self.state.all_losses += self.train_until_convergence(
                    (train_i, sampler_train_i, sampler_unlabelled_i),
                    (valid_i, sampler_valid_i, sampler_valid_unlabelled_i),
                    max_epochs=self.supervised_epochs_per_task,
                    description=f"Task {i} (Supervised)",
                )
                self.logger.debug(f"Size the state object: {getsizeof(self.state)}")

            # TODO: save the state during training.
            self.save()
            #  Evaluate on all tasks (as described above).
            cumul_loss = LossInfo(f"cumul_losses[{i}]")
            self.state.cumul_losses[i] = cumul_loss
            self.state.j = 0

            valid_log_dict = cumul_loss.to_log_dict()
            self.log({f"cumul_losses[{i}]": valid_log_dict})

        # TODO: Save the results to a json file.
        self.save(self.results_dir)
        # TODO: save the rest of the state.

        # make the plot of the losses (might not be useful, since we could also just do it in wandb).
        fig = self.make_loss_figure(self.all_losses, self.plot_sections)
        fig.savefig(self.plots_dir / "losses.png")

        if self.config.debug:
            fig.show()
            fig.waitforbuttonpress(10)


    def train_until_convergence(self, train_dataset: Tuple[Dataset, SubsetRandomSampler, SubsetRandomSampler],
                                valid_dataset: Tuple[Dataset, SubsetRandomSampler, SubsetRandomSampler],
                                max_epochs: int,
                                description: str = None,
                                patience: int = 3) -> Tuple[Dict[int, LossInfo], Dict[int, LossInfo]]:
        train_dataloader_labelled, train_dataloader_unlabelled  = self.get_dataloader(*train_dataset)
        valid_dataloader_labelled, valid_dataloader_unlablled  = self.get_dataloader(*valid_dataset)
        n_steps = len(train_dataloader_labelled)

        if self.config.debug_steps:
            from itertools import islice
            n_steps = self.config.debug_steps
            train_dataloader = islice(train_dataloader_labelled, 0, n_steps)  # type: ignore

        train_losses: Dict[int, LossInfo] = OrderedDict()
        valid_losses: Dict[int, LossInfo] = OrderedDict()

        valid_loss_gen = self.valid_performance_generator(valid_dataloader_labelled)

        best_valid_loss: Optional[float] = None
        counter = 0

        message: Dict[str, Any] = OrderedDict()
        for epoch in range(max_epochs):
            pbar = tqdm.tqdm(train_dataloader_labelled, train_dataloader_unlabelled, total=n_steps)
            desc = description or ""
            desc += " " if desc and not desc.endswith(" ") else ""
            desc += f"Epoch {epoch}"
            pbar.set_description(desc + " Train")

            for batch_idx, train_loss in enumerate(self.train_iter_semi_sup(pbar)):
                if batch_idx % self.config.log_interval == 0:
                    # get loss on a batch of validation data:
                    valid_loss = next(valid_loss_gen)
                    valid_losses[self.global_step] = valid_loss
                    train_losses[self.global_step] = train_loss

                    message.update(train_loss.to_pbar_message())
                    message.update(valid_loss.to_pbar_message())
                    pbar.set_postfix(message)

                    train_log_dict = train_loss.to_log_dict(verbose=True)
                    valid_log_dict = valid_loss.to_log_dict(verbose=True)
                    self.log({"Train": train_log_dict, "Valid": valid_log_dict})

            # perform a validation epoch.
            val_desc = desc + " Valid"
            val_loss_info = self.test(valid_dataloader_labelled, description=val_desc)
            val_loss = val_loss_info.total_loss

            if best_valid_loss is None or val_loss.item() < best_valid_loss:
                counter = 0
                best_valid_loss = val_loss.item()
            else:
                counter += 1
                print(f"Validation Loss hasn't decreased over the last {counter} epochs.")
                if counter == patience:
                    print(
                        f"Exiting at step {self.global_step}, as validation loss hasn't decreased over the last {patience} epochs.")
                    break
        return train_losses, valid_losses

    def test(self, dataloader: DataLoader, description: str = None, name: str = "Test") -> LossInfo:
        pbar = tqdm.tqdm(dataloader)
        desc = (description or "Test Epoch")

        pbar.set_description(desc)
        total_loss = LossInfo(name)
        message: Dict[str, Any] = OrderedDict()

        for batch_idx, loss in enumerate(self.test_iter(pbar)):
            total_loss += loss

            if batch_idx % self.config.log_interval == 0:
                message.update(total_loss.to_pbar_message())
                pbar.set_postfix(message)

        return total_loss

    def valid_performance_generator(self, periodic_valid_dataloader: DataLoader) -> Generator[LossInfo, None, None]:
        while True:
            for batch in periodic_valid_dataloader:
                data = batch[0].to(self.model.device)
                target = batch[1].to(self.model.device) if len(batch) == 2 else None
                yield self.test_batch(data, target)


    def train_iter_semi_sup(self, dataloader_sup: DataLoader, dataloader_unsup) -> Iterable[LossInfo]:
        self.model.train()
        for batch_sup, batch_unsup in zip(cycle(dataloader_sup),dataloader_unsup):
            data, target = self.preprocess(batch_sup)
            u, _ = self.preprocess(batch_unsup)
            yield self.train_batch_semi_sup(data,target,u)

    def train_batch_semi_sup(self, data: Tensor, target: Optional[Tensor], u: Tensor) -> LossInfo:
        self.model.optimizer.zero_grad()
        batch_loss_info = self.model.get_loss_semi(data, target, u)
        total_loss = batch_loss_info.total_loss
        total_loss.backward()
        self.model.optimizer.step()
        self.global_step += data.shape[0]
        return batch_loss_info


if __name__ == "__main__":
    from simple_parsing import ArgumentParser
    parser = ArgumentParser()
    parser.add_arguments(TaskIncrementalWithEWC, dest="experiment")
    
    args = parser.parse_args()
    experiment: TaskIncremental = args.experiment
    
    from main import launch
    launch(experiment)
