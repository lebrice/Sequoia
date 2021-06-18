from pathlib import Path
from dataclasses import dataclass
import json
from typing import Optional, Dict, Union, List, Tuple, Type
from sequoia.settings import Setting, Method, Results
from sequoia.common.config import Config
from .experiment import Experiment, parse_setting_and_method_instances
import sys
import shlex


@dataclass
class HPOSweep(Experiment):
    """ Experiment which launches an HPO Sweep using Orion.

    TODO: Maybe use this somewhere in main.py once we redesign the command-line API.
    """

    # Path to a json file containing the orion-formatted search space dictionary.
    # When `None` (by default), the result of `get_search_space` will be used instead.
    search_space_path: Optional[Path] = None
    # Path indicating where the pickle database will be loaded or be created.
    database_path: Path = Path("orion_db.pkl")
    # manual, unique identifier for this experiment. This should only really be used
    # when launching multiple different experiments that involve the same method and
    # the same exact setting configurations, but where some other aspect of the
    # experiment is changed.
    experiment_id: Optional[str] = None

    # Maximum number of runs to perform.
    max_runs: Optional[int] = 10

    def __post_init__(self):
        super().__post_init__()
        self.search_space: Dict = {}
        if self.search_space_path:
            with open(self.search_space_path, "r") as f:
                self.search_space = json.load(f)

    def launch(self, argv: Union[str, List[str]] = None, strict_args: bool = False):
        """Launch the experiment, using its attributes and possibly also using the
        provided command-line arguments.

        This differs from `Experiment.launch` in that this will actually launch a
        sequence of runs.

        Parameters
        ----------
        argv : Union[str, List[str]], optional
            [description], by default None
        strict_args : bool, optional
            [description], by default False

        Returns
        -------
        [type]
            [description]
        """
        if not (isinstance(self.setting, Setting) and isinstance(self.method, Method)):
            self.setting, self.method = parse_setting_and_method_instances(
                setting=self.setting,
                method=self.method,
                argv=argv,
                strict_args=strict_args,
            )
        assert isinstance(self.setting, Setting)
        assert isinstance(self.method, Method)
        self.setting.wandb = self.wandb

        # TODO: IDEA: It could actually be really cool if we created a list of
        # Experiment objects here, and just call their 'launch' methods in parallel,
        # rather than do the sweep logic in the Method class!
        best_params, best_objective = self.method.hparam_sweep(
            self.setting,
            search_space=self.search_space,
            database_path=self.database_path,
            experiment_id=self.experiment_id,
            max_runs=self.max_runs,
        )
        print(
            "Best params:\n"
            + "\n".join(f"\t{key}: {value}" for key, value in best_params.items())
        )
        print(f"Best objective: {best_objective}")
        return (best_params, best_objective)

    @classmethod
    def main(
        cls, argv: Union[str, List[str]] = None, strict_args: bool = False,
    ) -> List[Tuple[Dict, Results]]:
        """Launches this experiment from the command-line.

        First, we get the choice of method and setting using a first parser.
        Then, we parse the Setting and Method objects using the remaining args.

        Parameters
        ----------
        - argv : Union[str, List[str]], optional, by default None

            command-line arguments to use. When None (default), uses sys.argv.

        Returns
        -------
        List[Tuple[Dict, Results]]

            Best trial parameters and objective found during the sweep.

        """
        if argv is None:
            argv = sys.argv[1:]
        if isinstance(argv, str):
            argv = shlex.split(argv)
        _ = argv.copy()

        experiment: HPOSweep
        experiment, argv = cls.from_known_args(argv)

        setting: Optional[Type[Setting]] = experiment.setting
        method: Optional[Type[Method]] = experiment.method
        # config: Config = experiment.config

        if method is None or setting is None:
            raise RuntimeError(
                "Both `--setting` and `--method` must be set to run a sweep."
            )
        return experiment.launch(argv, strict_args=strict_args)


def main():
    HPOSweep.main()


if __name__ == "__main__":
    main()
