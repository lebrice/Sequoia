from typing import TYPE_CHECKING, List, Optional, Sequence, Union

import gym
import tqdm
from avalanche.benchmarks.scenarios import Experience
from avalanche.benchmarks.utils.avalanche_dataset import AvalancheDataset
from avalanche.benchmarks.utils.data_loader import (
    MultiTaskDataLoader, MultiTaskMultiBatchDataLoader)
from avalanche.training import default_logger
from avalanche.training.plugins import StrategyPlugin
from sequoia.settings.passive import (ClassIncrementalSetting,
                                      PassiveEnvironment)
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

if TYPE_CHECKING:
    from avalanche.training.plugins import StrategyPlugin

import torch
from avalanche.training.strategies.base_strategy import \
    BaseStrategy as _BaseStrategy
from sequoia.settings.passive.cl.class_incremental_setting import \
    ClassIncrementalTestEnvironment
from sequoia.settings.passive.cl.objects import Actions, Observations
from torch.utils.data import TensorDataset


class GatherDataset(StrategyPlugin):
    """ IDEA: A Plugin that accumulates the tensors from the env to create a "proper"
    Dataset to be used by the plugins.
    """
    def __init__(self):
        self.train_xs: List[Tensor] = []
        self.train_ys: List[Tensor] = []
        self.train_ts: List[Tensor] = []
        self.train_dataset: TensorDataset
        self.eval_xs: List[Tensor] = []
        self.eval_ys: List[Tensor] = []
        self.eval_ts: List[Tensor] = []
        self.eval_dataset: TensorDataset

    def after_forward(self, strategy, **kwargs):
        x, y, t = strategy.mb_x, strategy.mb_task_id, strategy.mb_y
        self.train_xs.append(x)
        self.train_ys.append(y)
        self.train_ts.append(t)
        return super().after_forward(strategy, **kwargs)

    def after_training_epoch(self, strategy, **kwargs):
        self.train_dataset = TensorDataset(
            torch.cat(self.train_xs), torch.cat(self.train_ys), torch.cat(self.train_ts)
        )
        self.train_xs.clear()
        self.train_ys.clear()
        self.train_ts.clear()
        return super().after_training_epoch(strategy, **kwargs)

    def after_eval_forward(self, strategy, **kwargs):
        x, y, t = strategy.mb_x, strategy.mb_task_id, strategy.mb_y
        self.eval_xs.append(x)
        self.eval_ys.append(y)
        self.eval_ts.append(t)
        return super().after_eval_forward(strategy, **kwargs)

    def after_eval_epoch(self, strategy, **kwargs):
        self.eval_dataset = TensorDataset(
            torch.cat(self.eval_xs), torch.cat(self.eval_ys), torch.cat(self.eval_ts)
        )
        self.eval_xs.clear()
        self.eval_ys.clear()
        self.eval_ts.clear()
        return super().after_eval_epoch(strategy, **kwargs)

    def train(self):
        return self.train_dataset

    def eval(self):
        return self.eval_dataset

    def after_training_exp(self, strategy: "BaseStrategy", **kwargs):
        """
        Compute importances of parameters after each experience.
        """
        if strategy.setting:
            strategy.experience.dataset = self.train_dataset
        return super().after_training_exp(strategy, **kwargs)

    def after_eval_exp(self, strategy: "BaseStrategy", **kwargs):
        """
        Compute importances of parameters after each experience.
        """
        if strategy.setting:
            strategy.experience.dataset = self.eval_dataset
        return super().after_eval_exp(strategy, **kwargs)


class BaseStrategy(_BaseStrategy):
    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        criterion,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: int = 1,
        device="cpu",
        plugins: Optional[Sequence["StrategyPlugin"]] = None,
        evaluator=default_logger,
        eval_every=-1,
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
        )
        self.setting: Optional[ClassIncrementalSetting] = None
        
        self.plugins.insert(0, GatherDataset())
        
        # IDEA: Construct the 'AvalancheDataset' by collecting the tensors when iterating
        # over the training env them during the first epoch! That way the plugins can
        # rely on there being a dataset!
        self.xs: List[Tensor] = []
        self.ys: List[Tensor] = []
        self.task_labels: List[Tensor] = []
        # self.dataset = strategy.experience.dataset

    @property
    def dataset_plugin(self) -> GatherDataset:
        return [plugin for plugin in self.plugins if isinstance(plugin, GatherDataset)][0]

    def train_dataset_adaptation(self, **kwargs):
        """ Initialize `self.adapted_dataset`. """
        self.adapted_dataset = self.experience.dataset
        if not isinstance(self.experience, gym.Env):
            self.adapted_dataset = self.adapted_dataset.train()

    def eval_dataset_adaptation(self, **kwargs):
        """ Initialize `self.adapted_dataset`. """
        self.adapted_dataset = self.experience.dataset
        if not isinstance(self.experience, gym.Env):
            self.adapted_dataset = self.adapted_dataset.eval()

    def make_train_dataloader(self, num_workers=0, shuffle=True, **kwargs):
        """
        Called after the dataset adaptation. Initializes the data loader.
        :param num_workers: number of thread workers for the data loading.
        :param shuffle: True if the data should be shuffled, False otherwise.
        """
        if self.setting:
            self.dataloader = self.setting.train_env
            # self.dataloader = self.setting.train_dataloader(
            #     batch_size=self.train_mb_size, num_workers=num_workers
            # )
        else:
            self.dataloader = MultiTaskMultiBatchDataLoader(
                self.adapted_dataset,
                oversample_small_tasks=True,
                num_workers=num_workers,
                batch_size=self.train_mb_size,
                shuffle=shuffle,
            )

    def make_eval_dataloader(self, num_workers=0, **kwargs):
        """
        Initializes the eval data loader.
        :param num_workers:
        :param kwargs:
        :return:
        """
        if self.setting:
            self.dataloader = self.setting.val_env
            # self.dataloader = self.setting.val_dataloader(
            #     batch_size=self.eval_mb_size, num_workers=num_workers
            # )
        else:
            self.dataloader = MultiTaskDataLoader(
                self.adapted_dataset,
                oversample_small_tasks=False,
                num_workers=num_workers,
                batch_size=self.eval_mb_size,
            )

    def training_epoch(self, **kwargs):
        """
        Training epoch.
        :param kwargs:
        :return:
        """
        if isinstance(self.dataloader, gym.Env):
            return self.training_epoch_gym_env(self.dataloader, **kwargs)
        return super().training_epoch(**kwargs)

    def training_epoch_gym_env(self, train_env: PassiveEnvironment, **kwargs):
        self.mb_it = 0
        episode = 0
        total_steps = 0
        # Only perform one 'episode' (one epoch).
        max_episodes = 1
        while not train_env.is_closed() and episode < max_episodes:
            observations: Observations = train_env.reset()
            done = False
            step = 0
            import tqdm

            with tqdm.tqdm(
                desc=f"Training epoch {self.epoch}/{self.train_epochs}"
            ) as pbar:
                while not done:
                    self.before_training_iteration(**kwargs)

                    self.optimizer.zero_grad()
                    self.loss = 0
                    # TODO: Do a forward pass for each task within the batch, to try and
                    # match what is done by their multi-task dataloader?

                    self.mb_x = observations.x
                    self.mb_task_id = observations.task_labels

                    self.mb_x = self.mb_x.to(self.device)

                    self.before_forward(**kwargs)
                    self.logits = self.model(self.mb_x)
                    # self.after_forward(**kwargs)

                    y_pred = self.logits.argmax(-1)
                    actions = Actions(y_pred=y_pred)

                    observations, rewards, done, info = train_env.step(actions)
                    step += 1
                    total_steps += 1

                    if not isinstance(done, bool):
                        assert False, done

                    self.mb_y = (
                        rewards.y.to(self.device) if rewards is not None else None
                    )
                    # TODO: Does `after_forward` need access to self.mb_y?
                    self.after_forward(**kwargs)
                    self.mb_it += 1
                    pbar.update()

                    self.loss = self.criterion(self.logits, self.mb_y)

                    self.before_backward(**kwargs)
                    self.loss.backward()
                    self.after_backward(**kwargs)

                    # Optimization step
                    self.before_update(**kwargs)
                    self.optimizer.step()
                    self.after_update(**kwargs)

                    self.after_training_iteration(**kwargs)

                    pbar.set_postfix(
                        {
                            "Episode": f"{episode}/{max_episodes}",
                            "step": f"{step}",
                            "total_steps": f"{total_steps}",
                            "loss": f"{self.loss.item()}",
                        }
                    )

            episode += 1

    def eval_epoch(self, **kwargs):
        if isinstance(self.experience, gym.Env):
            return self.eval_epoch_gym_env(test_env=self.experience, **kwargs)
        return super().eval_epoch(**kwargs)

    def test_epoch(self, test_env: ClassIncrementalTestEnvironment, **kwargs):
        return self.eval_epoch_gym_env(test_env)

    def eval_epoch_gym_env(self, test_env: ClassIncrementalTestEnvironment, **kwargs):
        self.mb_it = 0
        episode = 0
        total_steps = 0
        max_episodes = 1  # Only one 'episode' / 'epoch'.
        while not test_env.is_closed() and episode < max_episodes:
            observations: Observations = test_env.reset()
            done = False
            step = 0
            with tqdm.tqdm(desc="Eval epoch") as pbar:
                while not done:
                    self.before_eval_iteration(**kwargs)
                    self.mb_x = observations.x
                    self.mb_task_id = observations.task_labels

                    self.mb_x = self.mb_x.to(self.device)

                    self.before_eval_forward(**kwargs)
                    self.logits = self.model(self.mb_x)

                    y_pred = self.logits.argmax(-1)
                    actions = Actions(y_pred=y_pred)

                    observations, rewards, done, info = test_env.step(actions)
                    step += 1
                    pbar.update()
                    total_steps += 1

                    if not isinstance(done, bool):
                        assert False, done

                    self.mb_y = (
                        rewards.y.to(self.device) if rewards is not None else None
                    )
                    self.after_eval_forward(**kwargs)
                    self.mb_it += 1

                    self.loss = self.criterion(self.logits, self.mb_y)

                    self.after_eval_iteration(**kwargs)

                    pbar.set_postfix(
                        {
                            "Episode": f"{episode}/{max_episodes}",
                            "step": f"{step}",
                            "total_steps": f"{total_steps}",
                            "loss": f"{self.loss.item()}",
                        }
                    )
            episode += 1


__all__ = ["BaseStrategy"]
