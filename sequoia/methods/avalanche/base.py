import inspect
from dataclasses import dataclass, fields
from typing import ClassVar, Dict, Generic, List, Optional, Type, TypeVar, Union

import gym
import torch
import tqdm
from avalanche.benchmarks.scenarios import Experience
from avalanche.models import MTSimpleCNN, MTSimpleMLP, SimpleCNN, SimpleMLP
from avalanche.models.utils import avalanche_forward
from avalanche.training import EvaluationPlugin
from avalanche.training.plugins import StrategyPlugin
from avalanche.training.strategies import BaseStrategy
from avalanche.training.strategies.strategy_wrappers import default_logger
from gym import spaces
from gym.spaces.utils import flatdim
from sequoia.common.spaces import Image
from sequoia.methods import Method
from sequoia.settings.passive import (
    ClassIncrementalSetting,
    PassiveEnvironment,
    PassiveSetting,
)
from sequoia.settings.passive.cl.class_incremental_setting import (
    ClassIncrementalTestEnvironment,
)
from sequoia.settings.passive.cl.objects import Actions, Observations
from simple_parsing.helpers import choice
from simple_parsing.helpers.hparams import HyperParameters, log_uniform, uniform
from torch import nn, optim
from torch.nn import Module
from torch.optim import SGD
from torch.optim.optimizer import Optimizer

from .experience import SequoiaExperience

StrategyType = TypeVar("StrategyType", bound=BaseStrategy)


def environment_to_experience(
    env: PassiveEnvironment, setting: PassiveSetting
) -> Experience:
    """
    TODO: Somehow "convert"  the PassiveEnvironments (dataloaders) from Sequoia
    into an Experience from Avalanche.
    """
    return SequoiaExperience(env=env, setting=setting)


@dataclass
class AvalancheMethod(
    Method,
    HyperParameters,
    Generic[StrategyType],
    target_setting=ClassIncrementalSetting,
):
    """ Base class for all the Methods adapted from Avalanche. """

    # Name for the 'family' of methods, use to differentiate methods with the same name.
    family: ClassVar[str] = "avalanche"

    # The Strategy class to use for this Method. Subclasses have to add this property.
    strategy_class: ClassVar[Type[StrategyType]] = BaseStrategy

    # Class Variable to hold the types of models available as options for the `model`
    # field below.
    available_models: ClassVar[Dict[str, Type[nn.Module]]] = {
        "simple_cnn": SimpleCNN,
        "simple_mlp": SimpleMLP,
        "mt_simple_cnn": MTSimpleCNN,
        "mt_simple_mlp": MTSimpleMLP,
    }
    # Class Variable to hold the types of optimizers available for the `optimizer` field
    # below.
    available_optimizers: ClassVar[Dict[str, Type[Optimizer]]] = {
        "sgd": SGD,
        "adam": optim.Adam,
        "rmsprop": optim.RMSprop,
    }
    # Class variable to hold the types of loss functions available for the `criterion`
    # field below.
    available_criterions: ClassVar[Dict[str, Type[nn.Module]]] = {
        "cross_entropy_loss": nn.CrossEntropyLoss,
    }

    # The model.
    model: Union[Module, Type[Module]] = choice(available_models, default=SimpleCNN)
    # The optimizer to use.
    optimizer: Union[Optimizer, Type[Optimizer]] = choice(
        available_optimizers, default=optim.Adam
    )
    # The loss criterion to use.
    criterion: Union[Module, Type[Module]] = choice(
        available_criterions, default=nn.CrossEntropyLoss
    )
    # The train minibatch size. Defaults to 1.
    train_mb_size: int = 1
    # The number of training epochs. Defaults to 1.
    train_epochs: int = 1
    # The eval minibatch size. Defaults to 1.
    eval_mb_size: int = 1
    #  The device to use. Defaults to None (cpu).
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Plugins to be added. Defaults to None.
    plugins: Optional[List[StrategyPlugin]] = None
    # (optional) instance of EvaluationPlugin for logging and metric computations.
    evaluator: EvaluationPlugin = default_logger
    # The frequency of the calls to `eval` inside the training loop.
    # if -1: no evaluation during training.
    # if  0: calls `eval` after the final epoch of each training
    #     experience.
    # if >0: calls `eval` every `eval_every` epochs and at the end
    #     of all the epochs for a single experience.
    eval_every: int = -1
    # Learning rate of the optimizer.
    learning_rate: float = log_uniform(1e-6, 1e-2, default=1e-3)
    # L2 regularization term for the model weights.
    weight_decay: float = log_uniform(1e-12, 1e-3, default=1e-6)
    # Hidden size of the model, when applicable.
    hidden_size: int = uniform(128, 1024, default=512)
    # Number of workers of the dataloader. Defaults to 4.
    num_workers: int = 4

    def configure(self, setting: ClassIncrementalSetting) -> None:
        self.setting = setting
        self.model = self.create_model(setting).to(self.device)

        # Select the loss function to use.
        if not isinstance(self.criterion, nn.Module):
            self.criterion = self.criterion()

        self.optimizer = self.make_optimizer()
        # Actually initialize the strategy using the fields on `self`.
        self.cl_strategy: StrategyType = self.create_cl_strategy(setting)

    def create_cl_strategy(self, setting: ClassIncrementalSetting) -> StrategyType:
        strategy_constructor_params: List[str] = list(
            inspect.signature(self.strategy_class.__init__).parameters.keys()
        )
        cl_strategy_kwargs = {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name in strategy_constructor_params
        }
        return self.strategy_class(**cl_strategy_kwargs)

    def create_model(self, setting: ClassIncrementalSetting) -> Module:
        image_space: Image = setting.observation_space.x
        input_dims = flatdim(image_space)
        assert isinstance(
            setting.action_space, spaces.Discrete
        ), "assume a classification problem for now."
        num_classes = setting.action_space.n
        if isinstance(self.model, nn.Module):
            pass
        elif self.model is SimpleMLP:
            return self.model(
                input_size=input_dims,
                hidden_size=self.hidden_size,
                num_classes=num_classes,
            )
        elif self.model is MTSimpleMLP:
            return self.model(input_size=input_dims, hidden_size=self.hidden_size)
        elif self.model is SimpleCNN:
            return self.model(num_classes=num_classes)
        else:
            # These other models (MTSimpleCNN) don't seem to take any kwargs.
            return self.model()

    def make_optimizer(self) -> Optimizer:
        """ Creates the Optimizer. """
        optimizer_class = self.optimizer
        if isinstance(self.optimizer, Optimizer):
            optimizer_class = type(self.optimizer)
        return optimizer_class(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def fit(self, train_env: PassiveEnvironment, valid_env: PassiveEnvironment):
        train_exp = environment_to_experience(train_env, setting=self.setting)
        valid_exp = environment_to_experience(valid_env, setting=self.setting)
        self.cl_strategy.train(
            train_exp, eval_streams=[valid_exp], num_workers=self.num_workers
        )

    def get_actions(
        self,
        observations: ClassIncrementalSetting.Observations,
        action_space: gym.Space,
    ) -> ClassIncrementalSetting.Actions:
        observations = observations.to(self.device)
        # TODO: Perform inference with the model.
        with torch.no_grad():
            x = observations.x
            task_labels = observations.task_labels
            logits = avalanche_forward(self.model, x=x, task_labels=task_labels)
            y_pred = logits.argmax(-1)
            return self.target_setting.Actions(y_pred=y_pred)

    def on_task_switch(self, task_id: Optional[int]) -> None:
        if self.training:
            # No need to tell the cl_strategy, because we call `.train` which calls
            # `before_training_exp` with the current exp (the current task).
            pass
        else:
            # TODO: In Sequoia, the test 'epoch' goes through the sequence of tasks, not
            # necessarily in the same order as during training, while in Avalanche the
            # 'eval' occurs on a per-task basis.
            # TODO: There is a bug with task-incremental setting, where during testing
            # the algo might be tested on tasks it hasn't built an output layer for yet,
            # but building this layer requires calling `adaptation(dataset)` and this
            # dataset will be iterated on, which isn't great in the case of the test
            # env...
            # encountered before.
            # During test-time, there might be a task boundary, and we need to let the
            # cl_strategy and the plugins know.
            # TODO: Get this working, figure out what the plugins expect to retrieve
            # from the cl_strategy in this callback.
            pass

    def get_search_space(self, setting: ClassIncrementalSetting):
        return self.get_orion_space()

    def adapt_to_new_hparams(self, new_hparams):
        raise NotImplementedError(new_hparams)
        return super().adapt_to_new_hparams(new_hparams)


def test_epoch(strategy, test_env: ClassIncrementalTestEnvironment, **kwargs):
    strategy.is_training = False
    strategy.model.eval()
    strategy.model.to(strategy.device)

    # strategy.before_eval(**kwargs)

    # Data Adaptation
    # strategy.before_eval_dataset_adaptation(**kwargs)
    # strategy.eval_dataset_adaptation(**kwargs)
    # strategy.after_eval_dataset_adaptation(**kwargs)
    # strategy.make_eval_dataloader(**kwargs)

    # strategy.before_eval_exp(**kwargs)
    # strategy.eval_epoch(**kwargs)
    test_epoch_gym_env(strategy, test_env)
    # strategy.after_eval_exp(**kwargs)


def test_epoch_gym_env(strategy, test_env: ClassIncrementalTestEnvironment, **kwargs):
    strategy.mb_it = 0
    episode = 0
    strategy.experience = test_env
    total_steps = 0
    max_episodes = 1  # Only one 'episode' / 'epoch'.
    while not test_env.is_closed() and episode < max_episodes:
        observations: Observations = test_env.reset()
        done = False
        step = 0
        with tqdm.tqdm(desc="Eval epoch") as pbar:
            while not done:
                # strategy.before_eval_iteration(**kwargs)
                strategy.mb_x = observations.x
                strategy.mb_task_id = observations.task_labels

                strategy.mb_x = strategy.mb_x.to(strategy.device)
                # IDEA: Should probably return a random action whenever we have task
                # labels in the test loop the task id isn't a known one in the model:

                # strategy.before_eval_forward(**kwargs)

                strategy.logits = avalanche_forward(
                    model=strategy.model,
                    x=strategy.mb_x,
                    task_labels=strategy.mb_task_id,
                )

                y_pred = strategy.logits.argmax(-1)
                actions = Actions(y_pred=y_pred)

                observations, rewards, done, info = test_env.step(actions)
                step += 1
                pbar.update()
                total_steps += 1

                if not isinstance(done, bool):
                    assert False, done

                strategy.mb_y = (
                    rewards.y.to(strategy.device) if rewards is not None else None
                )
                # strategy.after_eval_forward(**kwargs)
                strategy.mb_it += 1

                strategy.loss = strategy.criterion(strategy.logits, strategy.mb_y)

                # strategy.after_eval_iteration(**kwargs)

                pbar.set_postfix(
                    {
                        "Episode": f"{episode}/{max_episodes}",
                        "step": f"{step}",
                        "total_steps": f"{total_steps}",
                        "loss": f"{strategy.loss.item()}",
                    }
                )
        episode += 1
