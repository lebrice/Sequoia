################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 01-12-2020                                                             #
# Author(s): Antonio Carta                                                     #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################
from collections import defaultdict
from typing import TYPE_CHECKING, List, Optional, Sequence, Union

import gym
from avalanche.benchmarks.scenarios import Experience
from avalanche.benchmarks.utils.avalanche_dataset import AvalancheDataset
from avalanche.benchmarks.utils.data_loader import (
    MultiTaskDataLoader, MultiTaskMultiBatchDataLoader)
from avalanche.training import default_logger
from avalanche.training.plugins import EvaluationPlugin
from gym import Env
from sequoia.settings.passive import (ClassIncrementalSetting,
                                      PassiveEnvironment, PassiveSetting)
from sequoia.settings.passive.cl.objects import Observations, Rewards
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import tqdm

if TYPE_CHECKING:
    from avalanche.training.plugins import StrategyPlugin

from avalanche.training.strategies.base_strategy import \
    BaseStrategy as _BaseStrategy
from sequoia.settings.passive.cl.class_incremental_setting import (
    ClassIncrementalSetting, ClassIncrementalTestEnvironment)
from sequoia.settings.assumptions.incremental import TestEnvironment

# TODO: Chat with Lorenzo Pellegrini for typing stuff
from sequoia.settings.passive.cl.objects import Observations, Rewards, Actions


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

    def make_train_dataloader(self, num_workers=0, shuffle=True, **kwargs):
        """
        Called after the dataset adaptation. Initializes the data loader.
        :param num_workers: number of thread workers for the data loading.
        :param shuffle: True if the data should be shuffled, False otherwise.
        """
        if isinstance(self.experience, gym.Env):
            self.dataloader = self.experience
            # TODO: Difference between train and valid?
        elif self.setting:
            self.dataloader = self.setting.train_dataloader(num_workers=num_workers)
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
        if isinstance(self.experience, gym.Env):
            self.dataloader = self.experience
            # TODO: Difference between train and valid?
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
            with tqdm.tqdm(desc=f"Training epoch {self.epoch}/{self.train_epochs}") as pbar:
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

                    self.mb_y = rewards.y.to(self.device) if rewards is not None else None
                    # TODO: Does `after_forward` need access to self.mb_y?
                    self.after_eval_forward(**kwargs)
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

                    pbar.set_postfix({
                        "Episode": f"{episode}/{max_episodes}",
                        "step": f"{step}",
                        "total_steps": f"{total_steps}",
                        "loss": f"{self.loss.item()}",
                    })

            episode += 1

    def eval_dataset_adaptation(self, **kwargs):
        """ Initialize `self.adapted_dataset`. """
        self.adapted_dataset = self.experience.dataset
        self.adapted_dataset = self.adapted_dataset.eval()

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
                    print(f"Episode {episode}, step={step}, total_steps={total_steps}")
                    self.before_eval_iteration(**kwargs)
                    self.mb_x = observations.x
                    self.mb_task_id = observations.task_labels

                    self.mb_x = self.mb_x.to(self.device)

                    self.before_eval_forward(**kwargs)
                    self.logits = self.model(self.mb_x)
                    self.after_eval_forward(**kwargs)  # Where to put this? Here?

                    y_pred = self.logits.argmax(-1)
                    actions = Actions(y_pred=y_pred)

                    observations, rewards, done, info = test_env.step(actions)
                    step += 1
                    pbar.update()
                    total_steps += 1

                    if not isinstance(done, bool):
                        assert False, done

                    self.mb_y = rewards.y.to(self.device) if rewards is not None else None
                    self.after_eval_forward(**kwargs)  # Or Here?
                    self.mb_it += 1

                    self.loss = self.criterion(self.logits, self.mb_y)

                    self.after_eval_iteration(**kwargs)

                    pbar.set_postfix({
                        "Episode": f"{episode}/{max_episodes}",
                        "step": f"{step}",
                        "total_steps": f"{total_steps}",
                        "loss": f"{self.loss.item()}",
                    })
            episode += 1

    def eval_epoch_active_dataloader(self, test_env: ClassIncrementalTestEnvironment, **kwargs):
        for self.mb_it, batch in enumerate(test_env):
            print(f"Test step {self.mb_it}")
            observations: Observations
            rewards: Optional[Rewards]
            if isinstance(batch, Observations):
                observations = batch
                rewards = None
            else:
                assert isinstance(batch, tuple) and len(batch) == 2
                observations, rewards = batch

            # (self.mb_x, self.mb_y, self.mb_task_id
            self.mb_x = observations.x
            self.mb_task_id = observations.task_labels

            self.mb_x = self.mb_x.to(self.device)
            self.mb_y = rewards.y.to(self.device) if rewards is not None else None

            self.before_eval_forward(**kwargs)
            self.logits = self.model(self.mb_x)
            self.after_eval_forward(**kwargs)

            y_pred = self.logits.argmax(-1)
            if rewards is None:
                # Need to send actions to the env before we can actually get the
                # associated Reward.
                # TODO: Assuming classification action for now.
                rewards = test_env.send(y_pred)
                assert False, (test_env.results, rewards.y, y_pred)

            assert False, (rewards.y, y_pred)

            self.mb_y = rewards.y.to(self.device)
            self.loss = self.criterion(self.logits, self.mb_y)

            self.after_eval_iteration(**kwargs)


__all__ = ["BaseStrategy"]
