import itertools
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import ClassVar, Dict, Optional, Type

import gym
import tqdm
import wandb
from gym.vector.utils import batch_space
from simple_parsing import field
from torch import Tensor

from sequoia.common.config import Config, WandbConfig
from sequoia.common.gym_wrappers.utils import IterableWrapper
from sequoia.common.metrics import Metrics, MetricsType
from sequoia.settings.base import Actions, Method, Setting
from sequoia.settings.base.results import Results
from sequoia.utils import add_prefix, get_logger
from sequoia.utils.utils import flag
from wandb.wandb_run import Run
from .base import AssumptionBase
from .iid_results import TaskResults

logger = get_logger(__file__)


@dataclass
class ContinualResults(TaskResults[MetricsType]):
    _runtime: Optional[float] = None
    _online_training_performance: Dict[int, MetricsType] = field(default_factory=dict)

    @property
    def online_performance(self) -> Dict[int, MetricsType]:
        """ Returns the online training performance.

        In SL, this is only recorded over the first epoch.

        Returns
        -------
        Dict[int, MetricType]
            a dictionary mapping from step number to the Metrics object produced at that
            step.
        """
        if not self._online_training_performance:
            return {}
        return self._online_training_performance

    @property
    def online_performance_metrics(self) -> MetricsType:
        return sum(self.online_performance.values(), Metrics())

    def to_log_dict(self, verbose: bool = False) -> Dict:
        log_dict = {}
        log_dict["Average Performance"] = super().to_log_dict(verbose=verbose)
        if self._online_training_performance:
            log_dict[
                "Online Performance"
            ] = self.online_performance_metrics.to_log_dict(verbose=verbose)
        return log_dict

    def summary(self, verbose: bool = False) -> str:
        s = StringIO()
        print(json.dumps(self.to_log_dict(verbose=verbose), indent="\t"), file=s)
        s.seek(0)
        return s.read()


@dataclass
class ContinualAssumption(AssumptionBase):
    """ Assumptions for Setting where the environments change over time. """
    # Which dataset to use.
    dataset: str

    known_task_boundaries_at_train_time: bool = flag(False)
    # Wether we get informed when reaching the boundary between two tasks during
    # training. Only used when `smooth_task_boundaries` is False.
    known_task_boundaries_at_test_time: bool = flag(False)
    # Wether we have sudden changes in the environments, or if the transition
    # are "smooth".
    smooth_task_boundaries: bool = flag(True)

    # Wether task labels are available at train time.
    # NOTE: Forced to True at the moment.
    task_labels_at_train_time: bool = flag(False)

    # Wether task labels are available at test time.
    task_labels_at_test_time: bool = flag(False)

    @dataclass(frozen=True)
    class Observations(AssumptionBase.Observations):
        task_labels: Optional[Tensor] = None

    @dataclass(frozen=True)
    class Actions(AssumptionBase.Actions):
        pass

    @dataclass(frozen=True)
    class Rewards(AssumptionBase.Rewards):
        pass

    # TODO: Move everything necessary to get ContinualRLSetting to work out of
    # Incremental and into this here. Makes no sense that ContinualRLSetting inherits
    # from Incremental, rather than this!

    Results: ClassVar[Type[ContinualResults]] = ContinualResults

    # Options related to Weights & Biases (wandb). Turned Off by default. Passing any of
    # its arguments will enable wandb.
    # NOTE: Adding `cmd=False` here, so we only create the args in `Experiment`.
    # TODO: Fix this up.
    wandb: Optional[WandbConfig] = field(default=None, compare=False, cmd=False)

    def main_loop(self, method: Method) -> ContinualResults:
        """ Runs a continual learning training loop, wether in RL or CL. """
        # TODO: Add ways of restoring state to continue a given run.
        if self.wandb and self.wandb.project:
            # Init wandb, and then log the setting's options.
            self.wandb_run = self.setup_wandb(method)
            method.setup_wandb(self.wandb_run)

        train_env = self.train_dataloader()
        valid_env = self.val_dataloader()

        logger.info(f"Starting training")
        method.set_training()
        self._start_time = time.process_time()

        method.fit(
            train_env=train_env, valid_env=valid_env,
        )
        train_env.close()
        valid_env.close()

        logger.info(f"Finished Training.")

        results = self.test_loop(method)

        if self.monitor_training_performance:
            results._online_training_performance = train_env.get_online_performance()

        logger.info(f"Resulting objective of Test Loop: {results.objective}")

        self._end_time = time.process_time()
        runtime = self._end_time - self._start_time
        results._runtime = runtime

        logger.info(f"Finished main loop in {runtime} seconds.")
        self.log_results(method, results)
        return results

    def test_loop(self, method: Method) -> "IncrementalAssumption.Results":
        """ WIP: Continual test loop.
        """
        test_env = self.test_dataloader()

        test_env: TestEnvironment

        was_training = method.training
        method.set_testing()

        try:
            # If the Method has `test` defined, use it.
            method.test(test_env)
            test_env.close()
            test_env: TestEnvironment
            # Get the metrics from the test environment
            test_results: Results = test_env.get_results()

        except NotImplementedError:
            logger.debug(
                f"Will query the method for actions at each step, "
                f"since it doesn't implement a `test` method."
            )
            obs = test_env.reset()

            # TODO: Do we always have a maximum number of steps? or of episodes?
            # Will it work the same for Supervised and Reinforcement learning?
            max_steps: int = getattr(test_env, "step_limit", None)

            # Reset on the last step is causing trouble, since the env is closed.
            pbar = tqdm.tqdm(itertools.count(), total=max_steps, desc="Test")
            episode = 0

            for step in pbar:
                if obs is None:
                    break
                # NOTE: The env might not be closed, while `obs` is actually still there.
                # if test_env.is_closed():
                #     logger.debug(f"Env is closed")
                #     break
                # logger.debug(f"At step {step}")

                # BUG: Need to pass an action space that actually reflects the batch
                # size, even for the last batch!

                # BUG: This doesn't work if the env isn't batched.
                action_space = test_env.action_space
                batch_size = getattr(
                    test_env, "num_envs", getattr(test_env, "batch_size", 0)
                )
                env_is_batched = batch_size is not None and batch_size >= 1
                if env_is_batched:
                    # NOTE: Need to pass an action space that actually reflects the batch
                    # size, even for the last batch!
                    obs_batch_size = obs.x.shape[0] if obs.x.shape else None
                    action_space_batch_size = (
                        test_env.action_space.shape[0]
                        if test_env.action_space.shape
                        else None
                    )
                    if (
                        obs_batch_size is not None
                        and obs_batch_size != action_space_batch_size
                    ):
                        action_space = batch_space(
                            test_env.single_action_space, obs_batch_size
                        )

                action = method.get_actions(obs, action_space)

                if test_env.is_closed():
                    break

                obs, reward, done, info = test_env.step(action)

                if done and not test_env.is_closed():
                    # logger.debug(f"end of test episode {episode}")
                    obs = test_env.reset()
                    episode += 1

            test_env.close()
            test_results: Results = test_env.get_results()

        if wandb.run:
            d = add_prefix(test_results.to_log_dict(), prefix="Test", sep="/")
            # d = add_prefix(test_metrics.to_log_dict(), prefix="Test", sep="/")
            # d["current_task"] = task_id
            wandb.log(d)

        # Restore 'training' mode, if it was set at the start.
        if was_training:
            method.set_training()

        return test_results
        # return test_results
        # if not self.task_labels_at_test_time:
        #     # TODO: move this wrapper to common/wrappers.
        #     test_env = RemoveTaskLabelsWrapper(test_env)

    def setup_wandb(self, method: Method) -> Run:
        """Call wandb.init, log the experiment configuration to the config dict.

        This assumes that `self.wandb` is not None. This happens when one of the wandb
        arguments is passed.

        Parameters
        ----------
        method : Method
            Method to be applied.
        """
        assert isinstance(self.wandb, WandbConfig)
        method_name: str = method.get_name()
        setting_name: str = self.get_name()

        if not self.wandb.run_name:
            # Set the default name for this run.
            run_name = f"{method_name}-{setting_name}"
            dataset = getattr(self, "dataset", None)
            if isinstance(dataset, str):
                run_name += f"-{dataset}"
            if getattr(self, "nb_tasks", 0) > 1:
                run_name += f"_{self.nb_tasks}t"
            self.wandb.run_name = run_name

        run: Run = self.wandb.wandb_init()
        run.config["setting"] = setting_name
        run.config["method"] = method_name
        run.config["method_full_name"] = method.get_full_name()
        run.summary["setting"] = self.get_name()
        if isinstance(self.dataset, str):
            run.summary["dataset"] = self.dataset
        run.summary["method"] = method.get_name()
        assert wandb.run is run
        return run

    def log_results(self, method: Method, results: Results, prefix: str = "") -> None:
        """
        TODO: Create the tabs we need to show up in wandb:
        1. Final
            - Average "Current/Online" performance (scalar)
            - Average "Final" performance (scalar)
            - Runtime
        2. Test
            - Task i (evolution over time (x axis is the task id, if possible))
        """
        logger.info(results.summary())

        if wandb.run:
            wandb.summary["method"] = method.get_name()
            wandb.summary["setting"] = self.get_name()
            dataset = getattr(self, "dataset", "")
            if dataset and isinstance(dataset, str):
                wandb.summary["dataset"] = dataset

            results_dict = results.to_log_dict()
            if prefix:
                results_dict = add_prefix(results_dict, prefix=prefix, sep="/")
            wandb.log(results_dict)

            # BUG: Sometimes logging a matplotlib figure causes a crash:
            # File "/home/fabrice/miniconda3/envs/sequoia/lib/python3.8/site-packages/plotly/matplotlylib/mplexporter/utils.py", line 246, in get_grid_style
            # if axis._gridOnMajor and len(gridlines) > 0:
            # AttributeError: 'XAxis' object has no attribute '_gridOnMajor'
            # Seems to be fixed by downgrading the matplotlib version to 3.2.2

            plots_dict = results.make_plots()
            if prefix:
                plots_dict = add_prefix(plots_dict, prefix=prefix, sep="/")
            wandb.log(plots_dict)
            # TODO: Finish the run here? Not sure this is right.
            # wandb.run.finish()

    @property
    def phases(self) -> int:
        """The number of training 'phases', i.e. how many times `method.fit` will be
        called.

        In the case of Continual and DiscreteTaskAgnostic, fit is only called once,
        with an environment that shifts between all the tasks. In Incremental, fit is
        called once per task, while in Traditional and MultiTask, fit is called once.
        """
        return 1


from gym.vector import VectorEnv
from sequoia.common.gym_wrappers.utils import EnvType


class TestEnvironment(gym.wrappers.Monitor, IterableWrapper[EnvType], ABC):
    """ Wrapper around a 'test' environment, which limits the number of steps
    and keeps tracks of the performance.
    """

    def __init__(
        self,
        env: EnvType,
        directory: Path,
        step_limit: int = 1_000,  # TODO: Remove this, use a dedicated wrapper for that.
        no_rewards: bool = False,
        config: Config = None,
        *args,
        **kwargs,
    ):
        super().__init__(env, directory, *args, **kwargs)
        # TODO: Need to stop re-creating the Monitor wrappers when we already have the list of envs
        # for each task!
        logger.info(f"Creating test env (Monitor) with log directory {self.directory}")
        self.step_limit = step_limit
        self.no_rewards = no_rewards
        self._steps = 0
        self.config = config
        # if wandb.run:
        #     wandb.gym.monitor()

    def step(self, action):
        self._before_step(action)
        # NOTE: Monitor wrapper from gym doesn't call `super().step`, so we have to
        # overwrite it here.
        observation, reward, done, info = IterableWrapper.step(self, action)
        done = self._after_step(observation, reward, done, info)
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._before_reset()
        observation = IterableWrapper.reset(self, **kwargs)
        self._after_reset(observation)
        return observation

    @abstractmethod
    def get_results(self) -> Results:
        """ Return how well the Method was applied on this environment.

        In RL, this would be based on the mean rewards, while in supervised
        learning it could be the average accuracy, for instance.

        Returns
        -------
        Results
            [description]
        """
        # TODO: Total reward over a number of steps? Over a number of episodes?
        # Average reward? What's the metric we care about in RL?
        rewards = self.get_episode_rewards()
        lengths = self.get_episode_lengths()
        total_steps = self.get_total_steps()
        return sum(rewards) / total_steps

    def step(self, action):
        # TODO: Its A bit uncomfortable that we have to 'unwrap' these here..
        # logger.debug(f"Step {self._steps}")
        action_for_stats = action.y_pred if isinstance(action, Actions) else action

        self._before_step(action_for_stats)

        if isinstance(action, Tensor):
            action = action.cpu().numpy()
        observation, reward, done, info = self.env.step(action)
        observation_for_stats = observation.x
        reward_for_stats = reward.y

        # TODO: Always render when debugging? or only when the corresponding
        # flag is set in self.config?
        try:
            if self.config and self.config.render and self.config.debug:
                self.render("human")
        except NotImplementedError:
            pass

        if isinstance(self.env.unwrapped, VectorEnv):
            done = all(done)
        else:
            done = bool(done)

        done = self._after_step(observation_for_stats, reward_for_stats, done, info)

        if self.get_total_steps() >= self.step_limit:
            done = True
            self.close()

        # Remove the rewards if they aren't allowed.
        if self.no_rewards:
            reward = None

        return observation, reward, done, info



TestEnvironment.__test__ = False
