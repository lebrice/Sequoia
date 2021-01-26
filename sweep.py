import json
from dataclasses import dataclass
from typing import Dict, Union

from orion.client import build_experiment, get_experiment
from orion.core.utils.exceptions import NoConfigurationError

from sequoia.common import Config
from sequoia.methods import BaselineMethod
from sequoia.settings import IIDSetting, Results, Setting
from sequoia.utils import compute_identity, dict_union
from sequoia.utils.logging_utils import get_logger

logger = get_logger(__file__)

@dataclass
class DummyMethod(BaselineMethod):
    def __init__(self, hparams=None, config=None, trainer_options=None, **kwargs):
        super().__init__(hparams=hparams, config=config, trainer_options=trainer_options, **kwargs)

    def get_experiment_name(self, setting: Setting) -> str:
        setting_hash = compute_identity(**setting.to_dict())
        # TODO: If the setting were to change, even very slightly, then the hash might be
        # very different! Do we really want to delete all previous points/runs while
        # developing Sequoia? 
        return f"{setting.get_name()}-{setting.dataset}_{setting_hash}"

    def get_search_space(self) -> Dict[str, Union[str, Dict]]:
        """ Gets the orion-formatted search space dictionary, mapping from hyper-parameter
        names to their priors.
        """
        space = self.hparams.get_orion_space()
        return space

    def hparam_sweep(self, setting: Setting, n_runs: int = None):
        """ IDEA: performs a hparam sweep using orion, changing the values in
        `self.hparams` progressively, and returning the best hparams found so far.
        """
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
            search_space_dict = self.get_search_space()
            experiment = build_experiment(
                name=experiment_name,
                space=search_space_dict,
                debug=self.config.debug,
                # storage={"type": "ephemeraldb"},
            )
            logger.info(f"Creating new experiment {experiment}")
        else:
            logger.info(f"Experiment {experiment} already exists.")

        best_hparams = self.hparams
        best_results = None
        while not experiment.is_done:
            trial = experiment.suggest()
            hparam_values = trial.params

            logger.info(f"Suggested values for this run:\n" + json.dumps(hparam_values, indent="\t"))

            current_hp_dict = self.hparams.to_dict()
            new_hp_dict = dict_union(current_hp_dict, hparam_values, recurse=True)
            new_hp = type(self.hparams).from_dict(new_hp_dict)
            
            self.hparams = new_hp
            self.configure(setting)
            assert self.model.hp is new_hp

            results: Results = setting.apply(self, config=self.config)
            run_results = [dict(
                    name=results.objective_name,
                    type='objective',
                    value=results.objective
                )
            ]
            # TODO: Orion works in a 'lower is better' fashion I think.

            experiment.observe(trial, run_results)

            if best_results is None:
                best_results = run_results
                best_hparams = self.hparams
            elif run_results > best_results:
                best_results = run_results
                best_hparams = self.hparams
        return best_hparams, best_perf



if __name__ == "__main__":
    from simple_parsing import ArgumentParser
    parser = ArgumentParser(description=__doc__)

    method = DummyMethod(config=Config(debug=True))
    setting = IIDSetting(dataset="mnist")
    best_hparams, best_perf = method.hparam_sweep(setting)
    print(f"Best hparams: {best_hparams}, best perf: {best_perf}")
    # results = setting.apply(method, config=Config(debug=True))


