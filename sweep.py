import json
from dataclasses import dataclass
from typing import Dict, Union, List, Optional, Mapping

from sequoia.common import Config
from sequoia.methods.baseline_method import BaselineMethod, BaselineModel
from sequoia.settings import IIDSetting, Results, Setting
from sequoia.utils import compute_identity, dict_union
from sequoia.utils.logging_utils import get_logger
import numpy as np
logger = get_logger(__file__)
import wandb

@dataclass
class DummyMethod(BaselineMethod):
    def __init__(self, hparams=None, config=None, trainer_options=None, **kwargs):
        super().__init__(hparams=hparams, config=config, trainer_options=trainer_options, **kwargs)

    def get_experiment_name(self, setting: Setting) -> str:
        """Gets a unique name for the experiment where `self` is applied to `setting`.

        This experiment name will be passed to `orion` when performing a run of
        Hyper-Parameter Optimization. 

        Parameters
        ----------
        - setting : Setting

            The `Setting` onto which this method will be applied. This method will be used when

        Returns
        -------
        str
            The name for the experiment.
        """
        setting_hash = compute_identity(size=5, **setting.to_dict())
        # TODO: If the setting were to change, even very slightly, then the hash might be
        # very different! Do we really want to delete all previous points/runs while
        # developing Sequoia? 
        return f"{setting.get_name()}-{setting.dataset}_{setting_hash}"

    def get_search_space(self) -> Mapping[str, Union[str, Dict]]:
        """ Gets the orion-formatted search space dictionary, mapping from hyper-parameter
        names to their priors.
        """
        space = self.hparams.get_orion_space()
        return space

    def hparam_sweep(self, setting: Setting, n_runs: int = None):
        """ IDEA: performs a hparam sweep using orion, changing the values in
        `self.hparams` progressively, and returning the best hparams found so far.
        """
        from orion.core.worker.trial import Trial
        from orion.client import build_experiment, get_experiment
        from orion.core.utils.exceptions import NoConfigurationError

        # Call 'configure', so that we create `self.model` at least once, which will
        # update
        # the hparams to be fully defined.
        self.configure(setting)
        assert self.hparams == self.model.hp
        experiment_name = self.get_experiment_name(setting)

        # Setting max epochs to 1, just to make runs a bit shorter.
        self.trainer_options.max_epochs = 1
        
        try:
            experiment = get_experiment(experiment_name, mode="w")
        except NoConfigurationError:
            # The experiment doesn't exist yet.
            logger.info(f"Creating new experiment with name {experiment_name}")
        else:
            logger.info(f"Experiment {experiment} already exists.")

        search_space_dict = self.get_search_space()
        experiment = build_experiment(
            name=experiment_name,
            space=search_space_dict,
            debug=self.config.debug,
            algorithms="BayesianOptimizer"
            # storage={"type": "pickleddb"},
        )

        previous_trials: List[Trial] = experiment.fetch_trials_by_status("completed")
        previous_hparams: List[BaselineModel.HParams] = [type(self.hparams).from_dict(trial.params) for trial in previous_trials]
        previous_results: List[Trial.Result] = [trial.results[0] for trial in previous_trials]
        
        best_results: Optional[Results] = None
        if previous_results:
            # (since Orion works in a 'lower is better' fashion)
            best_index = np.argmin([result.value for result in previous_results])
            best_hparams = previous_hparams[best_index]
            best_results = None
            best_objective = - previous_results[best_index].value
        else:
            best_hparams = self.hparams
            best_results = None
            best_objective = - np.inf

        logger.info(f"Best result encountered so far: {best_objective}")

        while not experiment.is_done:
            trial = experiment.suggest()
            hparam_values = trial.params

            logger.info(f"Suggested values for this run:\n" + json.dumps(hparam_values, indent="\t"))

            current_hp_dict = self.hparams.to_dict()
            new_hp_dict = dict_union(current_hp_dict, hparam_values, recurse=True)
            new_hp = type(self.hparams).from_dict(new_hp_dict)
            
            # Change the hyper-parameters, then reconfigure (which recreates the model).
            self.hparams = new_hp
            self.configure(setting)
            assert self.model.hp is new_hp

            result: Results = setting.apply(self, config=self.config)
            experiment.observe(trial, [
                dict(
                    name=result.objective_name,
                    type='objective',
                    # Note the minus sign, since lower is better in Orion.
                    value=-result.objective,
                )
            ])

            if best_results is None:
                best_results = result
                best_hparams = self.hparams
                best_objective = result.objective
            elif previous_results > best_results:
                best_results = result
                best_hparams = self.hparams
                best_objective = result.objective

            if wandb.run:
                wandb.run.finish()

            # FIXME: Remove this, just debugging stuff atm.
            # break
        return best_hparams, best_results


if __name__ == "__main__":
    from simple_parsing import ArgumentParser
    parser = ArgumentParser(description=__doc__)
    DummyMethod.add_argparse_args(parser, dest="method")
    parser.add_arguments(Config, dest="config")

    args = parser.parse_args()

    setting = IIDSetting(dataset="mnist")
    # setting: Setting = args.setting
    config: Config = args.config
    setting.config = config
    method = DummyMethod.from_argparse_args(args, dest="method")
    
    best_hparams, best_results = method.hparam_sweep(setting)

    print(f"Best hparams: {best_hparams}, best perf: {best_results}")
    # results = setting.apply(method, config=Config(debug=True))


