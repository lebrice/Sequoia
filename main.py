"""Runs an experiment, which consist in applying a Method to a Setting.


"""
import inspect
import json
import shlex
import traceback
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
from utils import Parseable, Serializable, get_logger

logger = get_logger(__file__)

logger.debug(f"Registered Settings: \n" + "\n".join(
    f"- {setting.get_name()}: {setting}" for setting in all_settings
))

logger.debug(f"Registered Methods: \n" + "\n".join(
    f"- {method.get_name()}: {method.target_setting} {method}" for method in all_methods
))

@dataclass
class Experiment(Parseable, Serializable):
    """ Applies a Method to an experimental Setting to obtain Results.

    When the `setting` is not set, calling `launch` on the
    `Experiment` will evaluate the chosen method on all "applicable" settings. 
    (i.e. lower in the Settings inheritance tree).

    When the `method` is not set, this will apply all applicable methods on the
    chosen setting.
    """
    # Which experimental setting to use. When left unset, will evaluate the
    # provided method on all applicable settings.
    setting: Optional[Union[str, Type[Setting]]] = choice(
        {setting.get_name(): setting for setting in all_settings},
        default=None,
    )

    # Which experimental method to use. When left unset, will evaluate all
    # compatible methods on the provided setting.
    # NOTE: Some methods can share the same name, for instance, 'baseline' may
    # refer to the ClassIncrementalMethod or TaskIncrementalMethod.
    # Therefore, the given `method` is a string (for example when creating this
    # class from the command-line) and there are multiple methods with the given
    # name, then the most specific method applicable for the given setting will
    # be used.
    method: Optional[Union[str, Type[Method]]] = choice(
        set(method.get_name() for method in all_methods),
        default=None,
    )

    def __post_init__(self):
        # The Setting subclass to be used.
        # When creating this object from the command-line, self.setting will
        # already be a Setting subclass (because of the 'choice' above), however
        # when created using the constructor directly, 
        self.setting_type: Optional[Type[Setting]] = None
        self.method_type:  Optional[Type[Method]] = None

        if not (self.setting or self.method):
            raise RuntimeError(
                "At least one of `setting` or `method` must be set!"
            )

        if self.setting is None:
            self.setting_type = None
        elif issubclass(self.setting, Setting):
            self.setting_type = self.setting
        elif isinstance(self.setting, str):
            # All settings must have a unique name.
            settings_with_that_name: List[Type[Setting]] = [
                setting for setting in all_settings
                if setting.get_name() == self.setting
            ]
            if not settings_with_that_name:
                raise RuntimeError(
                    f"No settings found with name '{self.setting}'!"
                    f"Settings available: \n" +
                    (
                        f"- {setting.get_name()}: {setting}\n" for setting in all_settings
                    ) 
                )
            elif len(settings_with_that_name) == 1:
                self.setting_type = settings_with_that_name[0]
            else:
                raise RuntimeError(
                    f"Error: There are multiple settings with the same name, "
                    f"which isn't allowed! (name: {self.setting}, culprits: "
                    f"{settings_with_that_name})"
                )

        if self.method is None:
            self.method_type = None
        if issubclass(self.method, Method):
            self.method_type = self.method_type
        elif isinstance(self.method, str):
            # Collisions in method names are allowed, and if it happens
            methods_with_that_name: List[Type[Method]] = [
                method for method in all_methods
                if method.get_name() == self.method
            ]
            if len(methods_with_that_name) == 1:
                self.method_type = methods_with_that_name[0]
            else:
                logger.warning(RuntimeWarning(
                    f"As there are multiple methods with the name {self.method}, "
                    f"this will try to use the most 'specialised' method with that "
                    f"name for the given setting. (potential methods: "
                    f"{methods_with_that_name}"
                ))
                # if self.setting_type:


    def launch(self, argv: Union[str, List[str]] = None) -> Optional[Results]:
        try:
            if issubclass(self.setting, Setting):
                self.setting = self.setting_type.from_args(argv)
            if self.method_type and not self.method:
                self.method = self.method_type.from_args(argv)

            if self.method and self.setting:
                if argv:
                    logger.warning(RuntimeWarning(f"Extra arguments:  {argv}"))
                return self.setting.apply(self.method)
            elif self.setting:
                # When the method isn't set, evaluate on all applicable methods.
                return self.setting.apply_all(argv)
            elif self.method:
                # When the setting isn't set, evaluate on all applicable settings.
                return self.method.apply_all(argv)

        except Exception as e:
            logger.error(f"Experiment crashed: {e}")
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
