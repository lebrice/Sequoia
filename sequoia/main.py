"""Sequoia - The Research Tree 

Used to run experiments, which consist in applying a Method to a Setting.
"""
from argparse import _SubParsersAction
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Type, Union

from simple_parsing import ArgumentParser
from simple_parsing.help_formatter import SimpleHelpFormatter
from simple_parsing.helpers import choice

import sequoia
from sequoia.common.config import Config
from sequoia.methods import get_all_methods
from sequoia.settings import all_settings
from sequoia.settings.base import Method, Results, Setting
from sequoia.utils import get_logger

# TODO: Fix all the `get_logger` to use __name__ instead of __file__.
logger = get_logger(__file__)


def main():
    """Adds all command-line arguments, parses the args, and runs the selected action."""
    parser = ArgumentParser(prog="sequoia", description=__doc__, add_dest_to_option_strings=False)
    parser.add_argument(
        "--version",
        action="version",
        version=sequoia.__version__,
        help="Displays the installed version of Sequoia and exits.",
    )

    command_subparsers = parser.add_subparsers(
        title="command",
        dest="command",
        description="Command to execute",
        parser_class=ArgumentParser,
        required=False,
    )

    add_run_command(command_subparsers)
    add_sweep_command(command_subparsers)
    add_info_command(command_subparsers)

    args = parser.parse_args()

    command: str = getattr(args, "command", None)
    if command is None:
        return parser.print_help()
    if command == "run":
        method_type: Type[Method] = args.method_type
        setting_type: Type[Setting] = args.setting_type
        # NOTE: There will most likely not be any conflict between these arguments here and those of
        # the setting, since they now each have their own subparser!
        # Therefore we don't really need the `dest` argument and all the associated messy prefixes.
        method: Method = method_type.from_argparse_args(args)
        setting: Setting = setting_type.from_argparse_args(args)
        config: Config = args.config
        return run(setting=setting, method=method, config=config)
    if command == "sweep":
        method_type: Type[Method] = args.method_type
        method: Method = method_type.from_argparse_args(args)
        return sweep(setting=args.setting, method=method, config=args.config)
    if command == "info":
        return info(component=args.component)


def add_run_command(command_subparsers: _SubParsersAction) -> None:
    run_parser = command_subparsers.add_parser(
        "run",
        description="Run an experiment on a given setting.",
        help="Run an experiment on a given setting.",
        add_dest_to_option_strings=False,
        formatter_class=SimpleHelpFormatter,
    )
    run_parser.add_arguments(Config, dest="config")
    add_args_for_settings_and_methods(run_parser)


def run(setting: Setting, method: Method, config: Config) -> Results:
    """Performs a single run, applying a method to a setting, and returns the results."""
    logger.debug("Setting:")
    # BUG: TypeError: __reduce_ex__() takes exactly one argument (0 given)
    try:
        logger.debug(setting.dumps_yaml())
    except TypeError:
        logger.debug(setting)
    logger.debug("Config:")
    logger.debug(config.dumps_yaml())
    logger.debug("Method")
    logger.debug(str(method))
    results = setting.apply(method, config=config)
    logger.debug("Results:")
    logger.debug(results.summary())
    return results


@dataclass
class SweepConfig(Config):
    """Configuration options for a HPO sweep."""

    # Path indicating where the pickle database will be loaded or be created.
    database_path: Path = Path("orion_db.pkl")
    # manual, unique identifier for this experiment. This should only really be used
    # when launching multiple different experiments that involve the same method and
    # the same exact setting configurations, but where some other aspect of the
    # experiment is changed.
    experiment_id: Optional[str] = None

    # Maximum number of runs to perform.
    max_runs: Optional[int] = 10

    # Which hyper-parameter optimization algorithm to use.
    hpo_algorithm: str = choice(
        {
            "random": "random",
            "bayesian": "BayesianOptimizer",
        },
        default="bayesian",
    )  # TODO: BayesianOptimizer does not support num > 1


def sweep(setting: Setting, method: Method, config: SweepConfig) -> Setting.Results:
    """Performs a Hyper-Parameter Optimization sweep, consisting in running the method
    on the given setting, each run having a different set of hyper-parameters.
    """
    print("Sweep!")
    logger.debug("Setting:")
    logger.debug(setting.dumps_yaml())
    logger.debug("Config:")
    logger.debug(config.dumps_yaml())
    logger.debug(f"Method: {method}")

    # TODO: IDEA: It could actually be really cool if we created a list of
    # Experiment objects here, and just call their 'launch' methods in parallel,
    # rather than do the sweep logic in the Method class!
    # TODO: Need to add these arguments again to the parser?
    best_params, best_objective = method.hparam_sweep(
        setting,
        database_path=config.database_path,
        experiment_id=config.experiment_id,
        max_runs=config.max_runs,
        hpo_algorithm=config.hpo_algorithm,
    )
    logger.info(
        "Best params:\n" + "\n".join(f"\t{key}: {value}" for key, value in best_params.items())
    )
    logger.info(f"Best objective: {best_objective}")
    return (best_params, best_objective)


def add_sweep_command(command_subparsers: _SubParsersAction) -> None:
    sweep_parser = command_subparsers.add_parser(
        "sweep",
        description="Run a hyper-parameter optimization sweep.",
        help="Run a hyper-parameter optimization sweep.",
        add_dest_to_option_strings=False,
    )
    sweep_parser.set_defaults(action=sweep)
    sweep_parser.add_arguments(SweepConfig, dest="config")
    add_args_for_settings_and_methods(sweep_parser)


def add_info_command(command_subparsers: _SubParsersAction) -> None:
    """Add commands to display some information about the settings or methods."""
    info_parser = command_subparsers.add_parser(
        "info",
        # NOTE: Not 100% sure what the difference is between help and description.
        description="Displays some information about a Setting or Method.",
        help="Displays some information about a Setting or Method.",
        add_dest_to_option_strings=False,
    )
    info_parser.set_defaults(**{"component": None})
    info_parser.set_defaults(action=lambda namespace: info(namespace.component))

    component_subparser = info_parser.add_subparsers(
        title="component",
        dest="component",
        description="Setting or Method to display more information about.",
        help="heyo",
        required=False,
    )

    for setting in all_settings:
        setting_name = setting.get_name()
        component_parser: ArgumentParser = component_subparser.add_parser(
            name=setting_name,
            description=f"Show more info about the {setting_name} setting.",
            help=get_help(setting),
            add_dest_to_option_strings=False,
        )
        component_parser.set_defaults(**{"component": setting})

    for method in get_all_methods():
        method_name = method.get_full_name()
        component_parser: ArgumentParser = component_subparser.add_parser(
            name=method_name,
            description=f"Show more info about the {method_name} method.",
            help=get_help(method),
            add_dest_to_option_strings=False,
        )
        component_parser.set_defaults(**{"component": method})


def info(component: Union[Type[Setting], Type[Method]] = None) -> None:
    """Prints some info about a given component (method class or setting class), or
    prints the list of available settings and methods.
    """
    if component is None:
        from sequoia.utils.readme import get_tree_string

        print(get_tree_string())

        # print("Registered Settings:")
        # for setting in all_settings:
        #     print(f"- {setting.get_name()}: {setting.get_path_to_source_file()}")

        print()
        print("Registered Methods:")
        print()
        for method in get_all_methods():
            src = method.get_path_to_source_file()
            print(f"- {method.get_full_name()}: {src}")

    else:
        # IDEA: Could colorize the tree with red or green depending on if the method is
        # applicable to the setting or not!
        help(component)


def get_help(component: Type[Setting]) -> str:
    """Returns the string to be passed as the 'help' argument to the parser."""
    # todo
    docstring = component.__doc__
    if not docstring:
        docstring = f"Help for class {component.__name__} (missing docstring)"
    # IDEA: Get the first two sentences, or a shortened version of the docstring,
    # whichever one is shorter.
    first_two_sentences = ". ".join(docstring.split(".")[:2]) + "."
    # shortened_docstring = textwrap.shorten(docstring, 150)
    # return min(shortened_docstring, first_two_sentences, key=len) + "(help)"
    # NOTE: Seems to be nicer in general to have two whole sentences, even if they are a bit longer.
    return first_two_sentences


# def get_description(command: str, setting: Type[Setting], method: Type[Method] = None) -> str:
#     """ Returns the text to be displayed right under the "usage" line in the command-line
#     when either
#     `sequoia run <setting> --help`
#     or
#     `sequoia run <setting> <method> --help` is invoked.
#     """
#     if command == "run":
#         if method is not None:
#             return f"Run an experiment consisting of applying method {method.get_full_name()} on the {setting.get_name()} setting. (desc.)"
#         else:
#             return f"Run an experiment in the {setting.get_name()} setting. (desc.)"


def add_args_for_settings_and_methods(command_subparser: ArgumentParser):
    """Adds a subparser for each Setting class and method subparsers for each of those.

    NOTE: Only adds subparsers for setting classes that have a non-empty 'available_datasets'
    attribute, so that choosing `Setting`, `SLSetting` or `RLSetting` isn't an option.

    This is used by the `sequoia run` and `sequoia sweep` commands.
    """
    # ===== RUN ========
    setting_subparsers = command_subparser.add_subparsers(
        title="setting_choice",
        description="choice of experimental setting",
        dest="setting_type",
        metavar="<setting>",
        required=True,
    )

    def key_fn(setting_class: Type[Setting]):
        return (
            len(setting_class.parents()),
            setting_class.__name__,
        )

    # Sort the settings so the actions come up in a nice order.
    for setting in sorted(all_settings, key=key_fn):
        setting_name = setting.get_name()

        # IDEA:
        if not getattr(setting, "available_datasets", {}):
            # Don't add a parser for this setitng, since it has no available datasets.
            # e.g.: Setting, SL, RL
            continue

        setting_parser: ArgumentParser = setting_subparsers.add_parser(
            setting_name,
            help=get_help(setting),
            description=f"Run an experiment in the {setting.get_name()} setting.",
            add_dest_to_option_strings=False,
            formatter_class=SimpleHelpFormatter,
        )
        setting_parser.set_defaults(**{"setting_type": setting})
        # NOTE: By removing the `dest` argument to `add_argparse_args, we're moving the place where
        # the setting's values are stored from 'setting' to `camel_case(setting_class.__name__).
        # Alternative would be to just assume that the settings are dataclasses and add arguments
        # for the setting at destination 'setting' as before.
        setting.add_argparse_args(parser=setting_parser)
        # setting_parser.add_arguments(setting, dest="setting")

        method_subparsers = setting_parser.add_subparsers(
            title="method",
            dest="method_name",
            metavar="<method>",
            description=f"which method to apply to the {setting_name} Setting.",
            required=True,
        )
        for method in setting.get_applicable_methods():
            method_name = method.get_full_name()
            method_parser: ArgumentParser = method_subparsers.add_parser(
                method_name,
                help=get_help(method),
                description=(
                    f"Run an experiment where the {method_name} method is "
                    f"applied to the {setting.get_name()} setting."
                ),
                formatter_class=SimpleHelpFormatter,
            )
            method_parser.set_defaults(method_type=method)
            # TODO: Could also pass the setting to the method's `add_argparse_args` so
            # that it gets to change its default values!
            # method.add_argparse_args_for_setting(
            #     parser=method_parser, setting=setting,
            # )
            method.add_argparse_args(parser=method_parser)


if __name__ == "__main__":
    main()
