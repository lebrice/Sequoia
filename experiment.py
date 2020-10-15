from dataclasses import InitVar, dataclass
from inspect import isabstract, isclass
from typing import Dict, List, Optional, Tuple, Type, TypeVar, Union

from simple_parsing import (ArgumentParser, choice, field, mutable_field,
                            subparsers)

from common.config import Config
from methods import MethodABC, all_methods
from settings import (ClassIncrementalResults, Results, Setting, SettingType,
                      all_settings)
from utils import Parseable, Serializable, get_logger
from utils.logging_utils import get_logger

logger = get_logger(__file__)

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
    setting: Optional[Union[str, Setting, Type[Setting]]] = choice(
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
    method: Optional[Union[str, MethodABC, Type[MethodABC]]] = choice(
        set(method.get_name() for method in all_methods),
        default=None,
    )
    
    # All the other configuration options, which are independant of the choice
    # of Setting or of Method, go in this next dataclass here! For example,
    # things like the log directory, wether Cuda is used, etc.
    config: Config = mutable_field(Config)

    def __post_init__(self):
        # if not (self.setting or self.method):
        #     raise RuntimeError(
        #         "At least one of `setting` or `method` must be set!"
        #     )
        if isinstance(self.setting, str):
            # All settings must have a unique name.
            settings_with_that_name: List[Type[Setting]] = [
                setting for setting in all_settings
                if setting.get_name() == self.setting
            ]
            if not settings_with_that_name:
                raise RuntimeError(
                    f"No settings found with name '{self.setting}'!"
                    f"Available settings : \n" + "\n".join(
                        f"- {setting.get_name()}: {setting}"
                        for setting in all_settings
                    )
                )
            elif len(settings_with_that_name) == 1:
                self.setting = settings_with_that_name[0]
            else:
                raise RuntimeError(
                    f"Error: There are multiple settings with the same name, "
                    f"which isn't allowed! (name: {self.setting}, culprits: "
                    f"{settings_with_that_name})"
                )

    def launch(self, argv: Union[str, List[str]] = None) -> Results:
        if isclass(self.setting) and issubclass(self.setting, Setting):
            self.setting = self.setting.from_args(argv)
            self.setting.config = self.config
        assert self.setting is None or isinstance(self.setting, Setting)

        method_name: Optional[str] = None
        if isinstance(self.method, str):
            # Collisions in method names should be allowed. If it happens,
            # we shoud use the right method for the given setting, if any.
            # There's also the special case where only a method string is given!
            # In that case, we'd have to loop over all the settings and get the
            # method applicable with the name 'method_name', as well as sort out
            # any conflicts..
            method_name = self.method

        if self.method and self.setting:
            # Evaluate a given method on a given setting.
            if method_name is not None:
                self.method = get_method_class_with_name(method_name, self.setting)
            if issubclass(self.method, MethodABC):
                self.method = self.method.from_args(argv)
            
            # Give the same Config to both the Setting and the Method.
            # TODO: Decide who should be holding what options from the config.
            self.method.config = self.config
            self.setting.config = self.config
            
            return self.setting.apply(self.method, config=self.config)

        elif self.setting is not None and self.method is None:
            # Evaluate all applicable methods on this setting.
            all_results: Dict[Type[MethodABC], Results] = {}

            for method_type in self.setting.get_applicable_methods():
                method = method_type.from_args(argv)
                results = self.setting.apply(method, config=self.config)
                all_results[method_type] = results

            logger.info(f"All results for setting of type {type(self)}:")
            logger.info({
                method_type.get_name(): (results if results else "crashed")
                for method_type, results in all_results.items()
            })
            return all_results
        
        elif self.method is not None and self.setting is None:
            # Evaluate this method on all applicable settings.
            all_results: Dict[Type[Setting], Results] = {}

            applicable_settings: List[Type[Setting]] = []
            if isinstance(self.method, str):
                # The name of the method to use if given, so we have to find all
                # settings that have an applicable method with that name.
                for setting in all_settings:
                    methods = setting.get_applicable_methods()
                    if any(m.get_name() == self.method for m in methods):
                        applicable_settings.append(setting)
            else:
                applicable_settings = self.method.get_applicable_settings()

            for setting_type in applicable_settings:
                # For each setting, if method_name was set, then we need to find
                # the right to use with that name from the list of applicable
                # methods.

                # Three possible cases: string, Method instance, or Method
                # subclass:
                assert isinstance(self.method, (str, Method)) or issubclass(self.method, MethodABC)
                if method_name is not None:
                    # We previously stored the name of the method in method_name  
                    self.method = get_method_class_with_name(method_name, setting_type)
                if isclass(self.method) and issubclass(self.method, MethodABC):
                    self.method = self.method.from_args(argv)

                setting = setting_type.from_args(argv)
                all_results[setting_type] = setting.apply(self.method, config=self.config)

            logger.info(f"All results for method of type {type(self.method)}:")
            logger.info({
                setting.get_name(): (results if results else "crashed")
                for setting, results in all_results.items()
            })
            return all_results

    @classmethod
    def main(cls, argv: Union[str, List[str]] = None) -> Union[Results, Dict[Type[Setting], Results], Dict[Type[MethodABC], Results]]:
        """Launches an experiment using the given command-line arguments.

        First, we get the choice of method and setting using a first parser.
        Then, we parse the Setting and MethodABC objects using the remaining args
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
        experiment: Experiment
        return experiment.launch(unused_args)


def get_method_class_with_name(method_name: str,
                               setting: Type[Setting] = None) -> Type[MethodABC]:
    potential_methods: List[Type[MethodABC]] = [
        method for method in all_methods
        if method.get_name() == method_name
    ]
    if setting:
        potential_methods = [
            m for m in potential_methods
            if m.is_applicable(setting)
        ]

    if not potential_methods:
        raise RuntimeError(
            f"Couldn't find any methods with name {method_name} "
            + (f"applicable on setting ({setting})!" if setting else "")
        )

    if len(potential_methods) == 1:
        return potential_methods[0]

    # Remove any method in the list who has descendants within the list.
    logger.warning(RuntimeWarning(
        f"As there are multiple methods with the name {method_name}, "
        f"this will try to use the most 'specialised' method with that "
        f"name for the given setting. (potential methods: "
        f"{potential_methods}"
    ))

    has_descendants: List[bool] = check_has_descendants(potential_methods)
    logger.debug(f"Method has descendants: {dict(zip(potential_methods, has_descendants))}")
    while any(has_descendants):
        indices_to_remove: List[int] = [
            i for i, has_descendant in enumerate(has_descendants) if has_descendant
        ]
        # pop the items in reverse index order so we don't mess up the list.
        for index_to_remove in reversed(indices_to_remove):
            potential_methods.pop(index_to_remove)
        has_descendants = check_has_descendants(potential_methods)

    assert len(potential_methods) > 0, "There should be at least one potential method left!"
    if len(potential_methods) == 1:
        return potential_methods[0]
    raise RuntimeError(
        f"There are more than one potential methods with name "
        f"{method_name} for setting of type {type(setting)}, and they aren't related "
        f"through inheritance! (potential methods: {potential_methods}"
    )

def check_has_descendants(potential_methods: List[Type[MethodABC]]) -> List[bool]:
    """Returns a list where for each method in the list, check if it has
    any descendants (subclasses of itself) also within the list.
    """
    def _has_descendant(method: Type[MethodABC]) -> bool:
        """ For a given method, check if it has any descendants within
        the list of potential methods.
        """
        return any(
            (issubclass(other_method, method) and
            other_method is not method)
            for other_method in potential_methods 
        )
    return [_has_descendant(method) for method in potential_methods]

