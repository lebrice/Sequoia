"""Runs an experiment, which consist in applying a Method to a Setting.


"""
import inspect
import shlex
from argparse import Namespace
from collections import OrderedDict
from dataclasses import InitVar, dataclass
from inspect import isabstract
from typing import Dict, List, Optional, Tuple, Type, TypeVar, Union

from methods import Method, MethodType, all_methods
from settings import (ClassIncrementalResults, Results, Setting, SettingType,
                      all_settings)
from simple_parsing import (ArgumentParser, choice, field, mutable_field,
                            subparsers)
from utils import Parseable, get_logger

logger = get_logger(__file__)

# Dict of all methods indexed by their names.
methods_dict: Dict[str, Type[Method]] = OrderedDict(
    (m.get_name(), m) for m in all_methods
)
# Dict of all settings indexed by their names.
settings_dict: Dict[str, Type[Setting]] = OrderedDict(
    (s.get_name(), s) for s in all_settings
)


@dataclass
class Experiment(Parseable):
    """ Applies a Method to an experimental Setting to obtain Results.

    When parsed through the command line, one of `setting_type` or `method_type`
    must be set. Alternatively, when creating an `Experiment` directly with the
    constructor, the `setting` or `method` arguments can be passed directly..

    When the `setting` or `setting_type` is not set, calling `launch` on the
    `Experiment` will evaluate the chosen method on all "applicable" settings 
    (i.e. lower in the Settings inheritance tree).

    When the `method` or `method_type` is not set, this will evaluate the
    chosen setting using all the methods (e.g. baselines) which are "applicable"
    to the chosen setting.
    """
    # Which experimental setting to use. When left unset, will evaluate the
    # provided method on all applicable settings.
    setting_type: Optional[Type[Setting]] = choice(
        settings_dict,
        default=None,
        alias="setting",
    )
    # Which experimental method to use. When left unset, will evaluate all
    # compatible methods with the provided setting.
    method_type:  Optional[Type[Method]] = choice(
        methods_dict,
        default=None,
        alias="method",
    )

    # We don't parse these directly from the command-line, we instead make it
    # possible to pass a Setting to the constructor.
    setting: InitVar[Optional[Setting]] = None
    method: InitVar[Optional[Method]] = None

    def __post_init__(self, setting: Optional[Setting] = None, method: Optional[Method] = None):
        self.setting: Optional[Setting] = setting
        self.method:  Optional[Method] = method
        # set the corresponding types when creating the Experiment manually
        # through the constructor.
        if self.setting and not self.setting_type:
            self.settings_type = type(self, setting)
        if self.method and not self.method_type:
            self.method_type = type(self.method)

    def launch(self, argv: Union[str, List[str]] = None) -> Optional[Results]:
        try:
            if not (self.setting_type or self.setting or self.method_type or self.method):
                raise RuntimeError(
                    f"Must specify at least either a setting or a method to be "
                    f"used!"
                )
            # Construct the Setting and Method from the args if they aren't set,
            # consuming the command-line arguments if necessary.
            if self.setting_type and not self.setting:
                self.setting, argv = self.setting_type.from_known_args(argv)
            if self.method_type and not self.method:
                self.method, argv = self.method_type.from_known_args(argv)

            if self.method and self.setting:
                if argv:
                    logger.warning(f"Extra arguments:  {argv}")
                return self.setting.apply(self.method)
            elif self.setting:
                # When the method isn't set, evaluate on all applicable methods.
                return self.setting.apply_all(argv)
            elif self.method:
                # When the setting isn't set, evaluate on all applicable settings.
                return self.method.apply_all(argv)

        except Exception as e:
            logger.error(f"Experiment crashed: {e}")
            import traceback
            traceback.print_exc()
            return None

    @classmethod
    def main(cls, argv: Union[str, List[str]] = None) -> Results:
        """Launches an experiment using the given command-line arguments.

        First, we get the choice of method and setting using a first parser.
        Then, we parse the Setting and Method objects using the remaining args
        with two other parsers.

        Parameters
        ----------
        - argv : Union[str, List[str]], optional, by default None

            command-line arguments to use. When None (default), uses sys.argv.

        Returns
        -------
        Results
            Results of the experiment.
        """
        experiment, unused_args = cls.from_known_args(argv)
        return experiment.launch(unused_args)

if __name__ == "__main__":
    results = Experiment.main()
    if results:
        # Experiment didn't crash, show results:
        print(f"Objective: {results.objective}")
        # print(f"Results: {results}")
        if isinstance(results, ClassIncrementalResults):
            print(f"task metrics:")
            for m in results.task_metrics:
                print(m)
