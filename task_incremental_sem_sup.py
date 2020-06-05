import tqdm
import torch
import wandb
import hashlib
import numpy as np
from sys import getsizeof
from torch import Tensor, nn
from torch.autograd import Variable
from itertools import repeat, cycle
from models.classifier import Classifier
from task_incremental import TaskIncremental
from dataclasses import dataclass
from torch.utils.data import Subset
from datasets.subset import ClassSubset
from common.losses import LossInfo
from datasets.ss_dataset import get_semi_sampler
from addons.curvature_analyser import Analyser
from collections import OrderedDict, defaultdict
from itertools import accumulate
from pathlib import Path
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, Dataset, TensorDataset
from typing import Dict, Iterable, List, Tuple, Union, Optional, Any
from typing import (Any, ClassVar, Dict, Generator, Iterable, List, Optional, Tuple, Type, Union)
from common.losses import LossInfo, TrainValidLosses
from simple_parsing import mutable_field, list_field
from common.task import Task
from task_incremental import get_supervised_accuracy
from utils.early_stopping import EarlyStoppingOptions, early_stopping
import logging


from tasks.simclr.simclr_task import SimCLRTask
from torchvision.transforms import Compose, Lambda, ToPILImage, ToTensor
try:
    from tasks.simclr.falr.config import HParams
    from tasks.simclr.falr.data import SimCLRAugment
    from tasks.simclr.falr.losses import SimCLRLoss
    from tasks.simclr.falr.models import Projector
except ImportError as e:
    print(f"Couldn't import the modules from the falr submodule: {e}")
    print("Make sure to run `git submodule init; git submodule update`")
    exit()


logger = logging.getLogger(__file__)

def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        lr = 1.0
    else:
        lr = current / rampup_length

    # print (lr)
    return lr

def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))

@dataclass
class TaskIncremental_Semi_Supervised(TaskIncremental):
    """ Evaluates the model in the same setting as the OML paper's Figure 3.
    """

    unsupervised_epochs_per_task: int = 0

    supervised_epochs_per_task: int = 10

    # Ratio of samples that have a corresponding label.
    ratio_labelled: float = 0.2
    # for supervised loss, the alpha parameter for the beta distribution from where the mixing lambda is drawn (used for ICT)
    mixup_sup_alpha: float = 0.
    #length of learning rate rampup in the beginning
    lr_rampup: int = 5
    #length of learning rate cosine rampdown (>= length of training): the epoch at which learning rate reaches to zero
    lr_rampdown_epochs: int = 0
    #If `True`, accuracy will be used as a measure of performance. Otherwise, the total validation loss is used. Defaults to False.
    use_accuracy_as_metric: bool= False
    #wether to apply simclr augment
    simclr_augment: bool = False
    #semi setup: 0 - only current task's unsupervised data, 1 - all tasks' unsupervised samples
    semi_setup_full: bool = 0

    def __post_init__(self):
        super().__post_init__()
        self.test_datasets: List[SubsetRandomSampler] = []
        self.train_samplers_labelled: List[SubsetRandomSampler] = []
        self.train_samplers_unlabelled: List[SubsetRandomSampler] = []
        self.valid_samplers_labelled: List[SubsetRandomSampler] = []
        self.valid_samplers_unlabelled: List[SubsetRandomSampler] = []
        self.test_samplers_labelled: List[SubsetRandomSampler] = []
        self.test_samplers_unlabelled: List[SubsetRandomSampler] = []

        self.epoch: Optional[int] = None
        self.epoch_length: Optional[int] = None
        self.batch_idx: Optional[int] = 0
        self.current_lr: Optional[float] = self.hparams.learning_rate

    def init_model(self) -> Classifier:
        self.logger.debug("init model")
        model = super().init_model()
        return model

    def adjust_learning_rate(self, optimizer, epoch, step_in_epoch, total_steps_in_epoch, epochs):
        current_lr = self.hparams.learning_rate   #self.current_lr
        initial_lr = 0.0 #self.hparams.learning_rate
        epoch = epoch + step_in_epoch / total_steps_in_epoch

        # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
        lr = linear_rampup(epoch, self.lr_rampup) * (current_lr - initial_lr) + initial_lr

        # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
        if self.lr_rampdown_epochs:
            assert self.lr_rampdown_epochs >= epochs
            lr *= cosine_rampdown(epoch, self.lr_rampdown_epochs)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    def load_datasets(self, tasks: List[Task]) -> None:
        """Create the train, valid and cumulative datasets for each task.

        Returns:
            List[List[int]]: The groups of classes for each task.
        """
        # download the dataset.
        self.train_dataset, self.test_dataset = self.dataset.load(data_dir=self.config.data_dir)

        # safeguard the entire training dataset.
        train_full_dataset = self.train_dataset
        test_full_dataset = self.test_dataset

        self.train_datasets.clear()
        self.valid_datasets.clear()
        self.test_datasets.clear()

        for i, task in enumerate(tasks):
            train = ClassSubset(train_full_dataset, task)
            train_size = int(0.8 * len(train))
            test_size = len(train) - train_size
            train_subset, valid_subset = torch.utils.data.random_split(train, [train_size, test_size])
            test = ClassSubset(test_full_dataset, task)

            sampler_train, sampler_train_unlabelled = get_semi_sampler(train.targets[train_subset.indices], p=self.ratio_labelled)
            sampler_valid, sampler_valid_unlabelled = get_semi_sampler(train.targets[valid_subset.indices], p=1.)
            sampler_test, sampler_test_unlabelled = get_semi_sampler(test.targets, p=1.)


            self.train_datasets.append(train_subset)
            self.train_samplers_labelled.append(sampler_train)
            self.train_samplers_unlabelled.append(sampler_train_unlabelled)

            self.valid_datasets.append(valid_subset)
            self.valid_samplers_labelled.append(sampler_valid)
            self.valid_samplers_unlabelled.append(sampler_valid_unlabelled)

            self.test_datasets.append(test)
            self.test_samplers_labelled.append(sampler_test)
            self.test_samplers_unlabelled.append(sampler_test_unlabelled)



        # Use itertools.accumulate to do the summation of validation datasets.
        self.valid_cumul_datasets = list(accumulate(self.valid_datasets))

        #for i, (train, test, valid, cumul) in enumerate(zip(self.train_datasets,
        #                                              self.test_dataset,
        #                                              self.valid_datasets,
        #                                              self.valid_cumul_datasets)):
        #    self.save_images(i, train, prefix="train_")
        #    self.save_images(i, test, prefix="test_")
        #    self.save_images(i, valid, prefix="valid_")
        #    self.save_images(i, cumul, prefix="valid_cumul_")

    def get_dataloaders(self,
                        dataset: Dataset,
                        sampler_labelled: SubsetRandomSampler,
                        sampler_unlabelled: SubsetRandomSampler) -> Tuple[DataLoader, DataLoader]:
        if sampler_labelled is not None and sampler_unlabelled is not None:
            loader_train_labelled = super().get_dataloader(dataset, sampler=sampler_labelled)
            loader_train_unlabelled = super().get_dataloader(dataset, sampler=sampler_unlabelled)
            return (loader_train_labelled, loader_train_unlabelled)
        else:
            loader = super().get_dataloader(dataset)
            return (loader, loader)


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
        self.log_dir.mkdir(parents=True, exist_ok=True)

        #if self.started or self.restore_from_path:
        #    self.logger.info(f"Experiment was already started in the past.")
        #    if not self.restore_from_path:
        #        self.restore_from_path = self.checkpoints_dir / "state.json"
        #    self.logger.info(f"Will load state from {self.restore_from_path}")
        #    self.load_state(self.restore_from_path)

        if self.done:
            self.logger.info(f"Experiment is already done.")
            # exit()

        if self.state.global_step == 0:
            self.logger.info("Starting from scratch!")
            self.state.tasks = self.create_tasks_for_dataset(self.dataset)
        else:
            self.logger.info(f"Starting from global step {self.state.global_step}")
            self.logger.info(f"i={self.state.i}, j={self.state.j}")

        self.tasks = self.state.tasks
        #self.save(save_model_weights=False)

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
            self.current_lr = self.hparams.learning_rate
            for param_group in self.model.optimizer.param_groups:
                param_group['lr'] = self.current_lr


            train_i = self.train_datasets[i]
            train_sampler_labeled_i = self.train_samplers_labelled[i]
            train_sampler_unlabelled_i = self.train_samplers_unlabelled[i]

            valid_i = self.valid_datasets[i]
            valid_sampler_labelled_i = self.valid_samplers_labelled[i]
            valid_sampler_unlabelled_i = self.valid_samplers_unlabelled[i]

            # If we are using a multihead model, we give it the task label (so
            # that it can spawn / reuse the output head for the given task).
            if self.multihead:
                prev_task = None if i==0 else self.tasks[i-1]
                classifier_head = None if i==0 else self.model.get_output_head(prev_task)
                self.on_task_switch(self.tasks[i], prev_task=prev_task, classifier_head = classifier_head,
                                    train_loader = self.get_dataloaders(
                                        dataset=train_i,
                                        sampler_labelled=train_sampler_labeled_i,
                                        sampler_unlabelled=train_sampler_unlabelled_i)[0])

            if self.state.j == 0:
                with self.plot_region_name(f"Learn Task {i}"):
                    # We only train (unsupervised) if there is at least one enabled
                    # auxiliary task and if the maximum number of unsupervised
                    # epochs per task is greater than zero.
                    self_supervision_on = any(task.enabled for task in self.model.tasks.values())

                    if self_supervision_on and self.unsupervised_epochs_per_task:
                        # Temporarily remove the labels.
                        with train_i.without_labels(), valid_i.without_labels():
                            # Un/self-supervised training on task i.
                            self.state.all_losses += self.train(
                                (train_i, train_sampler_labeled_i, train_sampler_unlabelled_i),
                                (valid_i, valid_sampler_labelled_i, valid_sampler_unlabelled_i),
                                epochs=self.unsupervised_epochs_per_task,
                                description=f"Task {i} (Unsupervised)",
                                temp_save_dir=self.f / f"task_{i}_unsupervised",
                            )
                    # Train (supervised) on task i.
                    # TODO: save the state during training?.
                    self.state.all_losses += self.train(
                        (train_i,train_sampler_labeled_i,train_sampler_unlabelled_i),
                        (valid_i,valid_sampler_labelled_i,valid_sampler_unlabelled_i),
                        epochs=self.supervised_epochs_per_task,
                        description=f"Task {i} (Supervised)",
                        use_accuracy_as_metric=self.use_accuracy_as_metric,
                        temp_save_dir=self.checkpoints_dir / f"task_{i}_supervised",
                    )
            # Save to the 'checkpoints' dir
            #self.save()
            # Evaluation loop:
            for j in range(self.state.j, self.n_tasks):
                if j == 0:
                    #  Evaluate on all tasks (as described above).
                    if self.state.cumul_losses[i] is not None:
                        logger.warning(
                            f"Cumul loss at index {i} should have been None "
                            f"but is {self.state.cumul_losses[i]}.\n"
                            f"This value be overwritten."
                        )
                    self.state.cumul_losses[i] = LossInfo(f"cumul_losses[{i}]")

                self.state.j = j
                train_j = self.train_datasets[j]
                train_sampler_labelled_j = self.train_samplers_labelled[j]
                train_sampler_unlabelled_j = self.train_samplers_unlabelled[j]

                test_j = self.test_datasets[j]
                test_sampler_labelled_j = self.test_samplers_labelled[j]
                test_sampler_unlabelled_j = self.test_samplers_unlabelled[j]

                train_dataloader_labelled, train_dataloader_unlabelled = self.get_dataloaders(
                    dataset=train_j,
                    sampler_labelled=train_sampler_labelled_j,
                    sampler_unlabelled=train_sampler_unlabelled_j,
                )
                test_dataloader_labelled, test_dataloader_unlablled = self.get_dataloaders(
                    dataset=test_j,
                    sampler_labelled=test_sampler_labelled_j,
                    sampler_unlabelled=test_sampler_unlabelled_j,
                )
                # Measure how linearly separable the representations of task j
                # are by training and evaluating a KNNClassifier on the data of task j.
                train_knn_loss, valid_knn_loss = self.test_knn(
                    train_dataloader_labelled,
                    test_dataloader_labelled,
                    description=f"KNN[{i}][{j}]"
                )
                self.log({
                    f"knn_losses[{i}][{j}]/train": train_knn_loss.to_log_dict(),
                    f"knn_losses[{i}][{j}]/valid": valid_knn_loss.to_log_dict(),
                })
                self.state.knn_losses[i][j] = valid_knn_loss
                accuracy = valid_knn_loss.metrics["KNN"].accuracy
                loss = valid_knn_loss.total_loss
                self.logger.info(f"knn_losses[{i}][{j}]/valid Accuracy: {accuracy:.2%}, loss: {loss}")

                if j <= i:
                    # If we have previously trained on this task:
                    if self.multihead:
                        self.on_task_switch(self.tasks[j])
                    loss_j = self.test(dataloader=test_dataloader_labelled, description=f"task_losses[{i}][{j}]")
                    self.state.cumul_losses[i] += loss_j
                    self.state.task_losses[i][j] = loss_j

                    self.log({f"task_losses/[{i}][{j}]": loss_j})


            # Save the state with the new metrics, but no need to save the
            # model weights, as they didn't change.
            #self.save(save_model_weights=False)

            self.state.j = 0
            cumul_loss = self.state.cumul_losses[i]
            self.log({f"cumul_losses[{i}]": cumul_loss})
            self.log({"task/currently_learned_task": i})

        # mark that we're done so we get right back here if we resume a
        # finished experiment
        self.state.i = self.n_tasks
        self.state.j = self.n_tasks
        #self.save(self.results_dir)  # Save to the 'results' dir.

        for i, cumul_loss in enumerate(self.state.cumul_losses):
            assert cumul_loss is not None, f"cumul loss at {i} should not be None!"
            cumul_valid_accuracy = get_supervised_accuracy(cumul_loss)
            self.logger.info(f"Cumul Accuracy [{i}]: {cumul_valid_accuracy}")
            if self.config.use_wandb:
                wandb.run.summary[f"Cumul Accuracy [{i}]"] = cumul_valid_accuracy

        from utils.plotting import maximize_figure
        # Make the forward-backward transfer grid figure.
        #grid = self.make_transfer_grid_figure(
        #    knn_losses=self.state.knn_losses,
        #    task_losses=self.state.task_losses,
        #    cumul_losses=self.state.cumul_losses
        #)
        #grid.savefig(self.plots_dir / "transfer_grid.png")

        # if self.config.debug:
        #     grid.waitforbuttonpress(10)

        # make the plot of the losses (might not be useful, since we could also just do it in wandb).
        # fig = self.make_loss_figure(self.all_losses, self.plot_sections)
        # fig.savefig(self.plots_dir / "losses.png")

        # if self.config.debug:
        #     fig.show()
        #     fig.waitforbuttonpress(10)
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
        total_loss.drop_tensors()
        return total_loss

    def train(self,
              train_dataset: Tuple[Dataset, SubsetRandomSampler, SubsetRandomSampler],
              valid_dataset: Tuple[Dataset, SubsetRandomSampler, SubsetRandomSampler],
              epochs: int,
              description: str = None,
              early_stopping_options: EarlyStoppingOptions = None,
              use_accuracy_as_metric: bool = None,
              temp_save_dir: Path = None) -> TrainValidLosses:
        """Trains on the `train_dataloader` and evaluates on `valid_dataloader`.

        Periodically evaluates on validation batches during each epoch, as well
        as doing a full pass through the validation dataset after each epoch.
        The weights at the point at which the model had the best validation
        performance are always re-loaded at the end.

        NOTE: If `early_stopping_options` is None, then the value from
        `self.config.early_stopping` is used. Same goes for
        `use_accuracy_as_metric`.

        NOTE: The losses are no logged to wandb, so you should log them yourself
        after this method completes.

        TODO: Add a way to resume training if it was previously interrupted.
        For instance, it might be useful to keep track of the number of epochs
        performed in the current task (for TaskIncremental)

        TODO: save/load the `all_losses` object to temp_save_file at a given
        interval during training, using a saver thread.


        Args:
            - train_dataloader (Union[Dataset, DataLoader]): Training dataset or
                dataloader.
            - valid_dataloader (Union[Dataset, DataLoader]): [description]
            - epochs (int): Number of epochs to train for.
            - description (str, optional): A description to use in the
                progressbar. Defaults to None.
            - early_stopping_options (EarlyStoppingOptions, optional): Options
                for configuring the early stopping hook. Defaults to None.
            - use_accuracy_as_metric (bool, optional): If `True`, accuracy will
                be used as a measure of performance. Otherwise, the total
                validation loss is used. Defaults to False.
            - temp_save_file (Path, optional): Path where the intermediate state
                should be saved/restored from. Defaults to None.

        Returns:
            TrainValidLosses: An object containing the training and validation
            losses during training (every `log_interval` steps) to be logged.
        """
        train_dataloader_labelled, train_dataloader_unlabelled = self.get_dataloaders(*train_dataset)
        if self.semi_setup_full:
            train_dataloader_unlabelled, _ = self.get_dataloaders(self.train_dataset, None, None)

        valid_dataloader_labelled, valid_dataloader_unlablled = self.get_dataloaders(*valid_dataset)
        early_stopping_options = early_stopping_options or self.config.early_stopping

        if use_accuracy_as_metric is None:
            use_accuracy_as_metric = self.config.use_accuracy_as_metric

        # The --debug_steps argument can be used to shorten the dataloaders.

        steps_per_epoch = len(train_dataloader_unlabelled) if len(train_dataloader_unlabelled) > len(
            train_dataloader_labelled) else len(train_dataloader_labelled)

        if self.config.debug_steps:
            from itertools import islice
            steps_per_epoch = self.config.debug_steps
            train_dataloader = islice(train_dataloader_labelled, 0, steps_per_epoch)  # type: ignore
        logger.debug(f"Steps per epoch: {steps_per_epoch}")

        # LossInfo objects at each step of validation
        validation_losses: List[LossInfo] = []
        # Container for the train and valid losses every `log_interval` steps.
        all_losses = TrainValidLosses()

        if temp_save_dir:
            temp_save_dir.mkdir(exist_ok=True, parents=True)

            all_losses_path = temp_save_dir / "all_losses.json"
            if all_losses_path.exists():
                all_losses = TrainValidLosses.load_json(all_losses_path)

            from itertools import count
            for i in count(start=1):
                loss_path = temp_save_dir / f"val_loss_{i}.json"
                if not loss_path.exists():
                    break
                else:
                    assert len(validation_losses) == (i - 1)
                    validation_loss = LossInfo.load_json(loss_path)
                    validation_losses.append(validation_loss)

            logger.info(f"Reloaded {len(validation_losses)} existing validation losses")
            logger.info(f"Latest step: {all_losses.latest_step()}.")

        # Get the latest step
        # NOTE: At the moment, will always be zero, but if we reload
        # `all_losses` from a file, would give you the step to start from.
        starting_step = all_losses.latest_step() or self.global_step
        starting_epoch = len(validation_losses) + 1

        if early_stopping_options:
            logger.info(f"Using early stopping with options {early_stopping_options}")

        # Hook to keep track of the best model.
        best_model_watcher = self.keep_best_model(
            use_acc=use_accuracy_as_metric,
            save_path=self.checkpoints_dir / "best_model.pth",
            # previous_losses=validation_losses,
        )
        next(best_model_watcher)

        # Hook to test for convergence.
        convergence_checker = early_stopping(
            options=early_stopping_options,
            use_acc=use_accuracy_as_metric,
            # previous_losses=validation_losses,
        )
        next(convergence_checker)

        # Hook for evaluating the validation performance periodically.
        valid_loss_gen = self.valid_performance_generator(valid_dataloader_labelled)

        # Message for the progressbar
        message: Dict[str, Any] = OrderedDict()
        # List to hold the length of each epoch (should all be the same length)
        epoch_lengths: List[int] = []

        for epoch in range(starting_epoch, epochs + 1):
            self.epoch = epoch
            self.epoch_length = len(train_dataloader_unlabelled)
            pbar = tqdm.tqdm(zip(cycle(train_dataloader_labelled), train_dataloader_unlabelled), total=steps_per_epoch)
            desc = description or ""
            desc += " " if desc and not desc.endswith(" ") else ""
            desc += f"Epoch {epoch}"
            pbar.set_description(desc + " Train")

            epoch_start_step = self.global_step
            for batch_idx, train_loss in enumerate(self.train_iter_semi_sup(pbar)):
                self.batch_idx = batch_idx
                train_loss.drop_tensors()

                if batch_idx % self.config.log_interval == 0:
                    # get loss on a batch of validation data:
                    valid_loss = next(valid_loss_gen)
                    valid_loss.drop_tensors()

                    all_losses[self.global_step] = (train_loss, valid_loss)

                    message.update(train_loss.to_pbar_message())
                    message.update(valid_loss.to_pbar_message())
                    pbar.set_postfix(message)

                    self.log({
                        "Train": train_loss,
                        "Valid": valid_loss,
                    })

            epoch_length = self.global_step - epoch_start_step
            epoch_lengths.append(epoch_length)

            # perform a validation epoch.
            val_desc = desc + " Valid"
            val_loss_info = self.test(valid_dataloader_labelled, description=val_desc)
            if temp_save_dir:
                val_loss_info.drop_tensors()
                # TODO: do this in the background saver thread.
                val_loss_info.save_json(temp_save_dir / f"val_loss_{i}.json")
                all_losses.save_json(temp_save_dir / f"all_losses.json")

            best_step = best_model_watcher.send(val_loss_info)
            logger.debug(f"Best step so far: {best_step}")

            best_epoch = (best_step - starting_step) // int(np.mean(epoch_length))
            logger.debug(f"Best epoch so far: {best_epoch}")

            converged = convergence_checker.send(val_loss_info)
            if converged:
                logger.info(f"Training Converged at epoch {epoch}. Best valid performance was at epoch {best_epoch}")
                break

        try:
            # Re-load the best weights
            best_model_watcher.send(None)
        except StopIteration:
            pass

        convergence_checker.close()
        best_model_watcher.close()
        valid_loss_gen.close()

        logger.info(f"Best step: {best_step}, best_epoch: {best_epoch}, ")
        all_losses.keep_up_to_step(best_step)

        return all_losses
    def train_until_convergence(self, train_dataset: Tuple[Dataset, SubsetRandomSampler, SubsetRandomSampler],
                                valid_dataset: Tuple[Dataset, SubsetRandomSampler, SubsetRandomSampler],
                                max_epochs: int,
                                description: str = None,
                                patience: int = 10) -> TrainValidLosses:
        train_dataloader_labelled, train_dataloader_unlabelled = self.get_dataloaders(*train_dataset)
        valid_dataloader_labelled, valid_dataloader_unlablled = self.get_dataloaders(*valid_dataset)
        n_steps = len(train_dataloader_unlabelled) if len(train_dataloader_unlabelled) > len(
            train_dataloader_labelled) else len(train_dataloader_labelled)

        if self.config.debug_steps:
            from itertools import islice
            n_steps = self.config.debug_steps
            train_dataloader = islice(train_dataloader_labelled, 0, n_steps)  # type: ignore

        all_losses = TrainValidLosses()
        # Get the latest step
        # NOTE: At the moment, will always be zero, but if we reload
        # `all_losses` from a file, would give you the step to start from.
        starting_step = all_losses.latest_step()

        valid_loss_gen = self.valid_performance_generator(valid_dataloader_labelled)

        best_valid_loss: Optional[float] = None
        counter = 0

        # Early stopping: number of validation epochs with increasing loss after
        # which we exit training.
        patience = patience or self.config.patience
        convergence_checker = self.check_for_convergence(patience=patience, use_acc=self.convegence_in_accuracy)
        next(convergence_checker)

        message: Dict[str, Any] = OrderedDict()
        for epoch in range(max_epochs):
            self.epoch = epoch
            self.epoch_length = len(train_dataloader_unlabelled)
            pbar = tqdm.tqdm(zip(cycle(train_dataloader_labelled), train_dataloader_unlabelled), total=n_steps)
            desc = description or ""
            desc += " " if desc and not desc.endswith(" ") else ""
            desc += f"Epoch {epoch}"
            pbar.set_description(desc + " Train")
            for batch_idx, train_loss in enumerate(self.train_iter_semi_sup(pbar)):
                self.batch_idx = batch_idx
                train_loss.drop_tensors()

                if batch_idx % self.config.log_interval == 0:
                    # get loss on a batch of validation data:
                    valid_loss = next(valid_loss_gen)
                    valid_loss.drop_tensors()

                    all_losses[self.global_step] = (train_loss, valid_loss)

                    message.update(train_loss.to_pbar_message())
                    message.update(valid_loss.to_pbar_message())
                    pbar.set_postfix(message)

                    train_log_dict = train_loss.to_log_dict()
                    valid_log_dict = valid_loss.to_log_dict()
                    self.log({"Train": train_log_dict, "Valid": valid_log_dict})

            # perform a validation epoch.
            val_desc = desc + " Valid"
            val_loss_info = self.test(valid_dataloader_labelled, description=val_desc)

            if epoch >= self.converge_after_epoch:
                converged = convergence_checker.send(val_loss_info)
                if converged:
                    convergence_checker.close()
                    break
        return all_losses
    def test_semi(self, dataloader_labelled: DataLoader, dataloader_unlabelled: DataLoader, description: str = None,
                  name: str = "Test") -> LossInfo:
        pbar = tqdm.tqdm(zip(cycle(dataloader_labelled), dataloader_unlabelled))
        desc = (description or "Test Epoch")

        pbar.set_description(desc)
        total_loss = LossInfo(name)
        message: Dict[str, Any] = OrderedDict()

        for batch_idx, loss in enumerate(self.test_iter_semi(pbar)):
            total_loss += loss

            if batch_idx % self.config.log_interval == 0:
                message.update(total_loss.to_pbar_message())
                pbar.set_postfix(message)

        return total_loss

    def preprocess_simclr(self, batch: Union[Tuple[Tensor], Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Optional[Tensor]]:

        data = batch[0].to(self.model.device)
        target = batch[1].to(self.model.device) if len(batch) == 2 else None  # type: ignore

        self.options = SimCLRTask.Options
        # Set the same values for equivalent hyperparameters
        self.options.image_size = data.shape[-1]
        self.options.double_augmentation = False

        self.augment = Compose([
            ToPILImage(),
            SimCLRAugment(self.options),
            Lambda(lambda tup: torch.stack([tup[0], tup[1]]))
        ])

        data = torch.cat([self.augment(x_i) for x_i in data.cpu()], dim=0)  # [2*B, C, H, W]
        target = torch.cat([ torch.stack([t,t]) for t in target.cpu()], dim=0) if target is not None else None
        return data.to(self.config.device), target.to(self.config.device)



    def test_iter_semi(self, dataloader: DataLoader) -> Iterable[LossInfo]:
        self.model.eval()
        for batch_sup, batch_unsup in dataloader:
            data, target = self.preprocess(batch_sup)
            u, _ = self.preprocess(batch_unsup)
            yield self.test_batch_semi(data, target, u)

    def test_batch_semi(self, data: Tensor, target: Tensor = None, u: Tensor = None) -> LossInfo:
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            loss = self.model.supervised_loss(data, target) + self.model.get_loss(u, None)
        if was_training:
            self.model.train()
        return loss

    def valid_performance_generator(self, periodic_valid_dataloader: DataLoader) -> Generator[LossInfo, None, None]:
        while True:
            for batch in periodic_valid_dataloader:
                data = batch[0].to(self.model.device)
                target = batch[1].to(self.model.device) if len(batch) == 2 else None
                yield self.test_batch(data, target)

    def train_iter_semi_sup(self, dataloader: DataLoader) -> Iterable[LossInfo]:
        self.model.train()
        for batch_sup, batch_unsup in dataloader:
            data, target = self.preprocess(batch_sup) if not self.simclr_augment else self.preprocess_simclr(batch_sup)
            u, _ = self.preprocess(batch_unsup) if not self.simclr_augment else self.preprocess_simclr(batch_unsup)
            yield self.train_batch_semi_sup(data, target, u)

    def preprocess_sup_mixup(self, x,y):
        from tasks.mixup import mixup_data_sup
        def mixup_criterion(y_a, y_b, lam):
            return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
        mixed_input, target_a, target_b, lam = mixup_data_sup(x, y, self.mixup_sup_alpha)
        mixed_input_var, target_a_var, target_b_var = Variable(mixed_input), Variable(target_a), Variable(target_b)
        loss_func = mixup_criterion(target_a_var, target_b_var, lam)
        return mixed_input_var, loss_func


    def train_batch_semi_sup(self, data: Tensor, target: Optional[Tensor], u: Tensor) -> LossInfo:
        if self.lr_rampdown_epochs > 0:
            self.current_lr = self.adjust_learning_rate(self.model.optimizer, self.epoch, self.batch_idx, self.epoch_length, self.supervised_epochs_per_task)
        self.log({'lr':self.current_lr})

        self.model.optimizer.zero_grad()
        loss_f = None
        data, target = self.model.preprocess_inputs(data, target)
        if self.mixup_sup_alpha>0:
            data, loss_f = self.preprocess_sup_mixup(data,target)

        batch_loss_info = self.model.supervised_loss(data, target, loss_f=loss_f) + self.model.get_loss(u, None)

        total_loss = batch_loss_info.total_loss
        total_loss.backward()
        self.model.optimizer_step(global_step=self.global_step,
                                  epoch=self.epoch,
                                  epoch_length=self.epoch_length,
                                  update_number=self.batch_idx)
        self.global_step += data.shape[0]
        return batch_loss_info


if __name__ == "__main__":
    from simple_parsing import ArgumentParser

    parser = ArgumentParser()
    parser.add_arguments(TaskIncremental_Semi_Supervised, dest="experiment")

    args = parser.parse_args()
    experiment: TaskIncremental = args.experiment

    from main import launch

    launch(experiment)
