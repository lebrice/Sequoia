""" Module used for launching an Experiment (applying a Method to one or more
Settings).
"""
import json
import shlex
import sys
from collections import defaultdict
from dataclasses import InitVar, dataclass
from inspect import isabstract, isclass
from typing import Dict, List, Optional, Tuple, Type, TypeVar, Union

from simple_parsing import (ArgumentParser, choice, field, mutable_field,
                            subparsers)

from sequoia.common.config import Config
from sequoia.methods import Method, all_methods
from sequoia.settings import (ClassIncrementalResults, Results, Setting, SettingType,
                      all_settings)
from sequoia.utils import Parseable, Serializable, get_logger
from sequoia.utils.logging_utils import get_logger

logger = get_logger(__file__)

@dataclass
class Experiment(Parseable, Serializable):
    """ Applies a Method to an experimental Setting to obtain Results.

    When the `setting` is not set, this will apply the chosen method on all of
    its "applicable" settings. (i.e. all subclasses of its target setting).

    When the `method` is not set, this will apply all applicable methods on the
    chosen setting.
    """
    # Which experimental setting to use. When left unset, will evaluate the
    # provided method on all applicable settings.
    setting: Optional[Union[Setting, Type[Setting]]] = choice(
        {setting.get_name(): setting for setting in all_settings},
        default=None,
        type=str,
    )

    # Which experimental method to use. When left unset, will evaluate all
    # compatible methods on the provided setting.
    # NOTE: Some methods can share the same name, for instance, 'baseline' may
    # refer to the ClassIncrementalMethod or TaskIncrementalMethod.
    # Therefore, the given `method` is a string (for example when creating this
    # class from the command-line) and there are multiple methods with the given
    # name, then the most specific method applicable for the given setting will
    # be used.
    method: Optional[Union[str, Method, Type[Method]]] = choice(
        set(method.get_name() for method in all_methods),
        default=None,
    )
    
    # All the other configuration options, which are independant of the choice
    # of Setting or of Method, go in this next dataclass here! For example,
    # things like the log directory, wether Cuda is used, etc.
    config: Config = mutable_field(Config)

    def __post_init__(self):
        if not (self.setting or self.method):
            raise RuntimeError("One of `setting` or `method` must be set!")

        # All settings have a unique name.
        if isinstance(self.setting, str):
            self.setting = get_class_with_name(self.setting, all_settings)

        # Each Method also has a unique name.
        if isinstance(self.method, str):
            self.method = get_class_with_name(self.method, all_methods)

        if self.setting is not None and self.method is not None:
            if not self.method.is_applicable(self.setting):
                raise RuntimeError(f"Method {self.method} isn't applicable to "
                                   f"setting {self.setting}!")
        
        assert self.setting is None or isinstance(self.setting, Setting) or issubclass(self.setting, Setting)
        assert self.method is None or isinstance(self.method, Method) or issubclass(self.method, Method)

    @staticmethod
    def run_experiment(setting: Union[Setting, Type[Setting]],
                       method: Union[Method, Type[Method]],
                       config: Config,
                       argv: Union[str, List[str]]=None,
                       strict_args: bool = False) -> Results:
        """ Launches an experiment, applying `method` onto `setting`
        and returning the corresponding results.
        
        This assumes that both `setting` and `method` are not None.
        This always returns a single `Results` object.
        
        If either `setting` or `method` are classes, then instances of these
        classes from the command-line arguments `argv`.
        
        If `strict_args` is True and there are leftover arguments (not consumed
        by either the Setting or the Method), a RuntimeError is raised.
        
        This then returns the result of `setting.apply(method)`.

        Parameters
        ----------
        argv : Union[str, List[str]], optional
            List of command-line args. When not set, uses the contents of
            `sys.argv`. Defaults to `None`.
        strict_args : bool, optional
            Wether to raise an error when encountering command-line arguments
            that are unexpected by both the Setting and the Method. Defaults to
            `False`.

        Returns
        -------
        Results
            
        """
        assert not (setting is None or method is None)
        
        if isinstance(setting, Setting) and isinstance(method, Method):
            return setting.apply(method, config=config)
        from simple_parsing import ArgumentParser, ConflictResolution

        # TODO: Should we raise an error if an argument appears both in the Setting
        # and the Method?
        parser = ArgumentParser(description=__doc__)
        
        if not isinstance(setting, Setting):
            assert issubclass(setting, Setting)
            setting.add_argparse_args(parser)
        if not isinstance(method, Method):
            assert issubclass(method, Method)
            method.add_argparse_args(parser)
        
        if strict_args:
            args = parser.parse_args(argv)
        else:
            args, unused_args = parser.parse_known_args(argv)
            if unused_args:
                logger.warning(UserWarning(f"Unused command-line args: {unused_args}"))
        
        if not isinstance(setting, Setting):
            setting = setting.from_argparse_args(args)
        if not isinstance(method, Method):
            method = method.from_argparse_args(args)
 
        assert isinstance(setting, Setting)
        assert isinstance(method, Method)
        assert isinstance(config, Config)

        return setting.apply(method, config=config)
        
    
    def launch(self,
               argv: Union[str, List[str]] = None,
               strict_args: bool = False,
               ) -> Results:
        """ Launches the experiment, applying `self.method` onto `self.setting`
        and returning the corresponding results.
        
        This differs from `main` in that this assumes that both `self.setting`
        and `self.method` are not None, and so this always returns a single
        `Results` object.
        
        NOTE: Internally, this is equivalent to calling `run_experiment`,
        passing in the `setting`, `method` and `config` arguments from `self`.
        
        Parameters
        ----------
        argv : Union[str, List[str]], optional
            List of command-line args. When not set, uses the contents of
            `sys.argv`. Defaults to `None`.
        strict_args : bool, optional
            Wether to raise an error when encountering command-line arguments
            that are unexpected by both the Setting and the Method. Defaults to
            `False`.

        Returns
        -------
        Results
            An object describing the results of applying Method `self.method` onto
            the Setting `self.setting`.
        """
        assert self.setting is not None
        assert self.method is not None
        assert self.config is not None
        return self.run_experiment(
            setting=self.setting,
            method=self.method,
            config=self.config,
            argv=argv,
            strict_args=strict_args,
        )

    @classmethod
    def main(cls,
             argv: Union[str, List[str]] = None,
             ) -> Union[Results,
                        Dict[Tuple[Type[Setting], Type[Method]], Results]]:
        """Launches one or more experiments from the command-line.

        First, we get the choice of method and setting using a first parser.
        Then, we parse the Setting and Method objects using the remaining args
        with two other parsers.

        Parameters
        ----------
        - argv : Union[str, List[str]], optional, by default None

            command-line arguments to use. When None (default), uses sys.argv.

        Returns
        -------
        Union[Results,
              Dict[Tuple[Type[Setting], Type[Method], Config], Results]]
            Results of the experiment, if only applying a method to a setting.
            Otherwise, if either of `--setting` or `--method` aren't set, this
            will be a dictionary mapping from
            (setting_type, method_type) tuples to Results.
        """
        
        if argv is None:
            argv = sys.argv[1:]
        if isinstance(argv, str):
            argv = shlex.split(argv)
        argv_copy = argv.copy()

        experiment: Experiment
        experiment, argv = cls.from_known_args(argv)
        
        setting: Optional[Type[Setting]] = experiment.setting
        method: Optional[Type[Method]] = experiment.method
        config: Config = experiment.config
    
        assert setting is None or issubclass(setting, Setting)
        assert method is None or issubclass(method, Method)

        if method is None and setting is None:
            raise RuntimeError(f"One of setting or method must be set.")
        
        if setting and method:
            # One 'job': Launch it directly.
            return experiment.launch(argv)

        # TODO: Maybe if everything stays exactly identical, we could 'cache'
        # the results of some experiments, so we don't re-run them all the time?
        all_results: Dict[Tuple[Type[Setting], Type[Method]], Results] = {}

        # The lists of arguments for each 'job'.
        method_types: List[Type[Method]] = []
        setting_types: List[Type[Setting]] = []
        run_configs: List[Config] = []

        if setting:
            logger.info(f"Evaluating all applicable methods on Setting {setting}.")
            method_types = setting.get_applicable_methods()
            setting_types = [setting for _ in method_types]

        elif method:
            logger.info(f"Applying Method {method} on all its applicable settings.")
            setting_types = method.get_applicable_settings()
            method_types = [method for _ in setting_types]

        # Create a 'config' for each experiment.
        # Use a log_dir for each run using the 'base' log_dir (passed
        # when creating the Experiment), the name of the Setting, and
        # the name of the Method.
        for setting_type, method_type in zip(setting_types, method_types):
            run_log_dir = config.log_dir / setting_type.get_name() / method_type.get_name()
            
            run_config_kwargs = config.to_dict()
            run_config_kwargs["log_dir"] = run_log_dir
            run_config = Config(**run_config_kwargs)

            run_configs.append(run_config)

        # TODO: Use submitit or something like it to launch jobs in parallel!
        # import submitit
        
        # Create one 'job' per setting-method combination:
        for setting_type, method_type, run_config in zip(setting_types,
                                                         method_types,
                                                         run_configs):
            # NOTE: Some methods might use all the values in `argv`, and some
            # might not, so we set `strict=False`.
            result = cls.run_experiment(
                setting=setting_type,
                method=method_type,
                config=run_config,
                argv=argv,
                strict_args=False,
            )
            logger.info(f"Results for setting {setting_type}, method {method_type}: {result.summary()}")
            all_results[(setting_type, method_type)] = result
        
        logger.info(f"All results: ")
        logger.info("\n" + json.dumps({
            f"{setting_type.get_name()} - {method_type.get_name()}": results.to_log_dict()
            for (setting_type, method_type), results in all_results.items()
        }, indent="\t"))
        return all_results


def get_class_with_name(class_name: str,
                        all_classes: Union[List[Type[Setting]], List[Type[Method]]],
                        ) -> Union[Type[Method], Type[Setting]]:
    potential_classes = [
        c for c in all_classes
        if c.get_name() == class_name
    ]
    # if target_class:
    #     potential_classes = [
    #         m for m in potential_classes
    #         if m.is_applicable(target_class)
    #     ]
    if len(potential_classes) == 1:
        return potential_classes[0]
    if not potential_classes:
        raise RuntimeError(
            f"Couldn't find any classes with name {class_name} in the list of "
            f"available classes {all_classes}!"
        )
    raise RuntimeError(
        f"There are more than one potential methods with name "
        f"{class_name}, which isn't supposed to happen! "
        f"(all_classes: {all_classes})"
    )

    # # Remove any method in the list who has descendants within the list.
    # logger.warning(RuntimeWarning(
    #     f"As there are multiple methods with the name {class_name}, "
    #     f"this will try to use the most 'specialised' method with that "
    #     f"name for the given setting. (potential methods: "
    #     f"{potential_classes}"
    # ))

    # has_descendants: List[bool] = check_has_descendants(potential_methods)
    # logger.debug(f"Method has descendants: {dict(zip(potential_classes, has_descendants))}")
    # while any(has_descendants):
    #     indices_to_remove: List[int] = [
    #         i for i, has_descendant in enumerate(has_descendants) if has_descendant
    #     ]
    #     # pop the items in reverse index order so we don't mess up the list.
    #     for index_to_remove in reversed(indices_to_remove):
    #         potential_classes.pop(index_to_remove)
    #     has_descendants = check_has_descendants(potential_classes)

    # assert len(potential_classes) > 0, "There should be at least one potential method left!"
    # if len(potential_classes) == 1:
    #     return potential_classes[0]
   
def check_has_descendants(potential_classes: List[Type[Method]]) -> List[bool]:
    """Returns a list where for each method in the list, check if it has
    any descendants (subclasses of itself) also within the list.
    """
    def _has_descendant(method: Type[Method]) -> bool:
        """ For a given method, check if it has any descendants within
        the list of potential methods.
        """
        return any(
            (issubclass(other_method, method) and
            other_method is not method)
            for other_method in potential_classes 
        )
    return [_has_descendant(method) for method in potential_classes]

