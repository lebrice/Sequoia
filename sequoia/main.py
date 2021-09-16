""" Sequoia - The Research Tree 

Used to run experiments, which consist in applying a Method to a Setting.
"""
from sequoia.settings.base import Setting, Method
from simple_parsing import ArgumentParser

# from sequoia.experiment import Experiment
from sequoia.common.config import Config
from sequoia.methods import get_all_methods
from sequoia.settings import all_settings
from sequoia.utils import get_logger
from sequoia.experiments import Experiment
from sequoia.settings.base import Results
from argparse import Namespace
from typing import Tuple
from simple_parsing.help_formatter import SimpleHelpFormatter


logger = get_logger(__file__)


def extract_args(args: Namespace) -> Tuple[Setting, Method, Config]:
    setting: Setting = args.setting
    method: Method = args.method
    config: Config = args.config
    return setting, method, config


def run(args: Namespace) -> None:
    setting, method, config = extract_args(args)
    print("Run!")
    print(f"Setting: {setting}")
    print(f"Method: {method}")
    print(f"Config: {config}")


def sweep(args: Namespace) -> None:
    setting, method, config = extract_args(args)
    print("Sweep!")
    print(f"Setting: {setting}")
    print(f"Method: {method}")
    print(f"Config: {config}")

def get_help(setting_or_method: Setting) -> str:
    return setting_or_method.__doc__ + " " + "(help)"

def get_description(setting_or_method: Setting) -> str:
    return setting_or_method.__doc__ + " " + "(description)"

def help_action(args: Namespace):
    logger.debug("Registered Settings: \n" + "\n".join(
        f"- {setting.get_name()}: {setting} ({setting.get_path_to_source_file()})" for setting in all_settings
    ))
    logger.debug("Registered Methods: \n" + "\n".join(
        f"- {method.get_full_name()}: {method} ({method.get_path_to_source_file()})" for method in get_all_methods()
    ))


def add_args_for_settings_and_methods(command_subparser: ArgumentParser):
    # ===== RUN ========
    setting_subparsers = command_subparser.add_subparsers(
        title="setting_choice",
        description="choice of experimental setting",
        dest="setting",
        metavar="<setting>",
        required=True,
    )
    for setting in all_settings:
        setting_name = setting.get_name()
        setting_parser: ArgumentParser = setting_subparsers.add_parser(
            setting.get_name(),
            help=get_help(setting),
            description=get_description(setting),
            add_dest_to_option_strings=False,
            formatter_class=SimpleHelpFormatter,
        )
        setting.add_argparse_args(parser=setting_parser, dest="setting")

        method_subparsers = setting_parser.add_subparsers(
            title="method",
            metavar="<method>",
            description=f"which method to apply to the {setting_name} Setting.",
            required=True,
        )
        for method in setting.get_applicable_methods():
            method_name = method.get_name()
            method_parser: ArgumentParser = method_subparsers.add_parser(
                method_name,
                help=get_help(method),
                description=get_description(method),
                add_dest_to_option_strings=False,
                formatter_class=SimpleHelpFormatter,
            )
            # TODO: Could also pass the setting to the method's `add_argparse_args`.
            method.add_argparse_args(parser=method_parser, dest="method")
            # method_parser.add_arguments(method, dest="method")


def main():
    parser = ArgumentParser(prog="sequoia", description=__doc__,
                            add_dest_to_option_strings=False)

    subparsers = parser.add_subparsers(
        title="command",
        dest="command",
        description="Command to execute",
        parser_class=ArgumentParser,
        required=True,  # assuming python > 3.7
    )
    setting_parser = subparsers.add_parser(
        "run",
        description="Run an experiment on a given setting.",
        help="Run an experiment on a given setting.",
        add_dest_to_option_strings=False,
        formatter_class=SimpleHelpFormatter,
    )
    setting_parser.set_defaults(action=run)
    setting_parser.add_arguments(Config, dest="config")
    add_args_for_settings_and_methods(setting_parser)

    sweep_parser = subparsers.add_parser(
        "sweep",
        description="Run a hyper-parameter optimization sweep.",
        help="Run a hyper-parameter optimization sweep.",
        add_dest_to_option_strings=False,
    )
    sweep_parser.set_defaults(action=sweep)
    sweep_parser.add_arguments(Config, dest="config")
    add_args_for_settings_and_methods(sweep_parser)

    info_parser = subparsers.add_parser(
        "info",
        description="Get some information on a given setting.",
        help="Print some information about a given setting or method.",
        add_dest_to_option_strings=False,
    )

    setting_subparsers = info_parser.add_subparsers(
        title="setting_choice",
        description="choice of experimental setting",
        dest="setting",
        required=True,
    )
    for setting in all_settings:
        setting_name = setting.get_name()
        setting_parser: ArgumentParser = setting_subparsers.add_parser(
            setting.get_name(),
            description=get_description(setting),
            help=get_help(setting),
            add_dest_to_option_strings=False,
        )
        from functools import partial
        setting_parser.set_defaults(action=partial(help, setting))
        # setting.add_argparse_args(parser=setting_parser, dest="setting")

    # method_subparsers = sweep_parser.add_subparsers(
    #     title="method_choice",
    #     description="choice of method?",
    #     required=True,
    # )
    # for method in all_methods:
    #     method_name = method.get_name()
    #     sweep_parser: ArgumentParser = method_subparsers.add_parser(
    #         method.get_name(),
    #         description=method.__doc__,
    #         add_dest_to_option_strings=False,
    #     )
    #     method.add_argparse_args(parser=sweep_parser, dest="method")
        
    #     setting_subparsers = sweep_parser.add_subparsers(
    #         title="setting",
    #         description=f"which setting to apply to the {method_name} method."
    #     )
    #     for setting in method.get_applicable_settings():
    #         setting_name = setting.get_name()
    #         run_parser: ArgumentParser = setting_subparsers.add_parser(
    #             setting_name,
    #             description=setting.__doc__,
    #             add_dest_to_option_strings=False,
    #         )
    #         setting.add_argparse_args(parser=run_parser, dest="setting")
    #         # setting_parser.add_arguments(setting, dest="setting")
    
    args = parser.parse_args()
    
    action = args.action

    return action(args)

    
        
    # subparsers.add_parser("help", )    
    
    
    # results = Experiment.main()
    # if results:
    #     print("\n\n EXPERIMENT IS DONE \n\n")
    #     # Experiment didn't crash, show results:
    #     print(f"Results: {results}")


if __name__ == "__main__":
    main()
